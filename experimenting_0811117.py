import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import pi, exp, cos, sin, sqrt, cos, sin, log, log10, ceil
from pylab import savefig
from scipy.integrate import quad
import cmath as cm
from scipy.integrate import simps
from scipy.special import spence
import time
import math
import sys

import warnings
warnings.filterwarnings("ignore")

plots = 1  # Do you want to output plots? 1 for yes

inspect_single = 0

class timeError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)


# =================================================================#
# DEFINING FUNCTIONS
# =================================================================#

# Cube root which will return the real root for negative numbers
def cr(x):
	if x < 0:
		return -(-x) ** (1. / 3)
	else:
		return x ** (1. / 3)


# creates and array with 'points' number of points for every order of magnitude
def scalesArray(initial, final, points):
	imag = ceil(log10(initial))
	fmag = int(log10(final))

	span = int(fmag - imag)

	if span >= 0:

		if int(log10(initial)) == log10(initial):
			ipoints = 1
		else:
			ipoints = points

		if int(log10(final)) == log10(final):
			fpoints = 0
		else:
			fpoints = points - 1

		mu = 0.0 * np.linspace(0., 1., ipoints + span * (points - 1) + fpoints)
		mu[0:ipoints] = np.linspace(initial, 10 ** imag, ipoints)

		j = 1
		while j <= span:
			temp = np.linspace(10 ** (imag + j - 1), 10 ** (imag + j), points)
			mu[(ipoints + (j - 1) * (points - 1)): (ipoints + j * (points - 1))] = temp[1:]
			j = j + 1

		if fpoints != 0:
			temp = np.linspace(10 ** (fmag), final, fpoints + 1)
			mu[-fpoints: len(mu)] = temp[1:]

		return mu
	else:
		return np.linspace(initial, final, points)


def find_nearest_idx(array, value):
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
		return idx - 1
	else:
			return idx

def T_BH(mss):
	return 1 / (8 * pi * G * mss)

# =================================================================#
# CLASSES
# =================================================================#

class Particle:
	mass = 0
	dof = 0
	Ap = 0
	Gamma_s = 1
	t_mu = 0

	def __init__(self, m, d):
		self.mass = m
		self.dof = d
		self.Ap = -1. * self.dof * (self.Gamma_s / (2 * pi)) * 1. / (6 * (8. * pi * G) ** 2)


# =================================================================#
# DEFINING INITIAL PARAMETERS
# =================================================================#
Mpl = 1.220910 * 10 ** 19  # (GeV) Planck mass
G = 1 / Mpl ** 2  # (GeV)^-2

mbh0 = 5. * 10 ** 32  # (GeV) initial black hole mass
T0 = T_BH(mbh0)  # (GeV) initial BH temp
t0 = 0.001 / (6.6 * 10 ** (-25))  # 1/(6.6*10**(-25)) is 1 second
nsm0 = 50  # number of 'massless' SM d.o.f.
nDM = 50  # number of massive DM d.of.
Gamma_s = 1.





# -----------------------------------------------------------------#
SM_light = Particle(0, 96)
SM_heavy = Particle(100,22)

# -----------------------------------------------------------------#
muVals = scalesArray(1000, 10**7, 20)  # (GeV) mass of DM #np.array([T0]) #
print muVals
rhoDMs = np.zeros(len(muVals))
rhoSMHs = np.zeros(len(muVals))
rhoSMLs = np.zeros(len(muVals))
ratios = np.zeros(len(muVals))
t_endVals = np.zeros(len(muVals))
j = 0
# -----------------------------------------------------------------#

# -----------------------------------------------------------------#
# Analytic expressions

def m_(t, t_init, Apar, m0):
	return cr(3. * Apar) * cr(t - t_init + (m0 ** 3) / (3. * Apar))

try:
	for mu in muVals:
		DM = Particle(mu, 96)

		ParticleList = [SM_light, DM]

		AsmL = SM_light.Ap
		AsmH = SM_heavy.Ap
		Asm = AsmH + AsmL
		ADM = DM.Ap
		Atot = Asm + ADM

		#-----------------------------#
		#  CASE 1: mu > heavy SM mass #
		#-----------------------------#
		if mu > SM_heavy.mass:	# if SML -> SMH -> DM
			
			# heavy SM vals
			mbh_H = 1 / (8 * pi * G * SM_heavy.mass)			# BH mass when SM_h produced
			t_smH = t0 + (mbh_H ** 3 - mbh0 ** 3) / (3. * AsmL)	# time when SM_h produced
			if t_smH < t0:
				t_smH = t0
				mbh_H = mbh0
					
			# DM vals
			mbh_mu = 1 / (8 * pi * G * mu)
			tmu = t_smH + (mbh_mu ** 3 - mbh_H ** 3) / (3. * Asm)
			if tmu <= t0:
				tmu = t0
				mbh_mu = mbh0


			def m(t): # gives mass at any time, accounting for all epochs
				if t <= t_smH:
					result = m_(t, t0, AsmL, mbh0)
				elif t <= tmu:
					result = m_(t, t_smH, Asm, mbh_H)
				else:
					result = m_(t, tmu, Atot, mbh_mu)
				return result

			t_end = tmu + ((10*Mpl)**3 - mbh_mu ** 3) / (3. * Atot)

			tSML = scalesArray(t0, t_end, 1000)
			m_arr = np.array([m(i) for i in tSML])
	
			def rhoSMLdot(y, t):
				dydt = - 4. * 2. / (3 * t) * y - AsmL / m(t) ** 2 * (t0 / t) ** 2
				return dydt

			def rhoSMHdot(y, t):
				dydt = - 3. * 2. / (3 * t) * y - AsmH / m(t) ** 2 * (t0 / t) ** 2
				return dydt

			def rhoDMdot(y, t):
				dydt = - 3. * 2. / (3 * t) * y - ADM / m(t) ** 2 * (t0 / t) ** 2
				return dydt	
		
			rhoSML0 = 0
			rhoSMH0 = 0		
			rhoDM0 = 0
		
			tSMH = scalesArray(t_smH, t_end, 1000)
			tDM = scalesArray(tmu, t_end, 1000)
			
			solSML = odeint(rhoSMLdot, rhoSML0, tSML)
			solSMH = odeint(rhoSMHdot, rhoSMH0, tSMH)
			solDM = odeint(rhoDMdot, rhoDM0, tDM)
		
			rhoDMs[j] = solDM[-1] if not np.isnan(solDM[-1]) else rhoDMs[j-1]
			rhoSMHs[j] = solSMH[-1] if not np.isnan(solSMH[-1]) else rhoSMHs[j-1]
			rhoSMLs[j] = solSML[-1] if not np.isnan(solSML[-1]) else rhoSMLs[j-1]
			ratios[j] = rhoDMs[j] / (rhoSMHs[j] + rhoSMLs[j] + rhoDMs[j])
			

			print 'rhoDM= ' + str(rhoDMs[j])
			print 'rhoSMH= ' + str(rhoSMHs[j])
			print 'rhoSML= ' + str(rhoSMLs[j])
			print 'ratio= ' + str(ratios[j])
			print solSMH

			if inspect_single:
					plt.plot(tDM, solDM, marker='.')
					plt.xlabel('t')
					plt.ylabel('rho_DM')
					plt.show()
					
					plt.plot(tDM, solSML, marker='.')
					plt.xlabel('t')
					plt.ylabel('rhoSML')
					plt.show()

			print 'tmu=' + str(tmu) + ', t_end = ' + str(t_end)  

	#		if ratios[j] < 10**(-5):
	#			print 'T0 = ' + str(T0) + ', mu = ' + str(mu) + ', mu/T0 = ' + str(mu/T0)
	#			print 't0 = ' + str(t0) + ', tmu = ' + str(tmu)
	#			raise timeError(tmu - t0)

			t_endVals[j] = t_end * (6.6 * 10 ** (-25))
			j = j + 1
		#----------------------------#
		'''


		def m(tm, tnu):
			Aeff = AsmL
			if tm >= tnu:
				Aeff += ADM
			# GET t_sm_heavy and account for it here
			result = m_(tm, t0, Aeff)
			return result

		# T = mu

		tmu = t0
		t_end = 0

		if T0 < mu:
			mbh_mu = 1 / (8 * pi * G * mu)  # mass of black hole at T = mu
			tmu = t0 + (mbh_mu ** 3 - mbh0 ** 3) / (3. * AsmL)
			t_end = tmu - (mbh_mu ** 3) / (3. * Atot)
		else:
			t_end = t0 - (mbh0 ** 3) / (3. * Atot)

		# m = [cr(3*A)*cr(x-t0+(mbh0**3)/(3*A)) for x in t]




		# Plotting m vs t

		t = scalesArray(t0, t_end, 1000)  # 10.98**19*t0

	#	tmu_index = find_nearest_idx(t, tmu)
	#	print('tmu index', tmu_index)

		m_arr = np.array([m(i, tmu) for i in t])

		if plots:
			plt.plot(t, m_arr, marker='.')

			plt.xlabel('t')
			plt.ylabel('m_BH')

			plt.show()


		# ===============================================#
		#				  ODEs							 #
		# ===============================================#


		def rhoSMdot(y, t):
			dydt = - 4. * 2. / (3 * t) * y - AsmL / m(t, tmu) ** 2 * (t0 / t) ** 2
			return dydt


		def rhoDMdot(y, t):
			dydt = - 3. * 2. / (3 * t) * y - ADM / m(t, tmu) ** 2 * (t0 / t) ** 2
			return dydt


		rhoSM0 = 0
		rhoDM0 = 0

		tDM = scalesArray(t0 if t0>tmu else tmu, t_end, 1000)

		solSM = odeint(rhoSMdot, rhoSM0, t)
		solDM = odeint(rhoDMdot, rhoDM0, tDM)

	
		# -------------------------------------#
		# Testing for matter-radiation equality
		# -------------------------------------#
		SM_index = 0
		DM_index = 0 if t0 > tmu else tmu_index

		MR_eq_index = -1
	
		for i in range(0,len(t)):
			if i < DM_index:
				ratio_MR = #rhoBH / solSM[i]
			else:
				ratio_MR = (solDM[i] # + rhoBH[i]) / solSM[i]
			if ratio_MR <= 1:
				MR_eq_index = i
				break
			
		# -------------------------------------#
		#mat-rad eq

		print t0, tmu, t_end
		print solSM[-1]
		print solDM[-1]

		if plots == 1:
			plt.semilogx(t, solSM, marker='.')
			plt.xlabel('t')
			plt.ylabel('rho_SM')
			plt.show()

			plt.plot(tDM, solDM, marker='.')
			plt.xlabel('t')
			plt.ylabel('rho_DM')
			plt.show()

		ratios[j] = solDM[-1] / (solSM[-1] + solDM[-1])
		t_endVals[j] = t_end * (6.6 * 10 ** (-25))
		j = j + 1
		'''
except timeError as error:
	print "tmu - t0 = " + str(error.value)
#	sys.exit()

	
print rhoDMs
print ratios

if not inspect_single:
	# DEBUGGING
	plt.loglog(muVals / T0, rhoDMs, marker='.')
	plt.xlabel('$\mu / T_0$  (mass of DM d.o.f.)', fontsize=15)
	plt.ylabel(r'$\rho_{DM}$', fontsize=20)
	plt.show()

	plt.loglog(muVals / T0, rhoSMHs, marker='.')
	plt.xlabel('$\mu / T_0$  (mass of DM d.o.f.)', fontsize=15)
	plt.ylabel(r'$\rho_{SMh}$', fontsize=20)
	plt.show()

	plt.loglog(muVals / T0, rhoSMLs, marker='.')
	plt.xlabel('$\mu / T_0$  (mass of DM d.o.f.)', fontsize=15)
	plt.ylabel(r'$\rho_{SMl}$', fontsize=20)
	plt.show()
	###

	plt.loglog(muVals / T0, ratios, marker='.')
	plt.xlabel('$\mu / T_0$  (mass of DM d.o.f.)', fontsize=15)
	plt.ylabel(r'$\frac{\rho_{DM}}{\rho_{DM}+\rho_{SM}}$', fontsize=20)
	plt.title('Mbh0 = ' + str(mbh0) + ', nSM = nDM = ' + str(SM_light.dof) + ',   $t_0$ = ' + str(t0*(6.6 * 10 ** (-25))) + 's')
	plt.show()

	plt.loglog(muVals / T0, t_endVals, marker='.')

	plt.show()

	print (np.log(ratios[-1]) - np.log(ratios[-10])) / (np.log(muVals[-1]) - np.log(muVals[-10]))

