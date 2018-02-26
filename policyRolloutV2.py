from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import copy as cp
import random
import numpy as np
import os, sys
import csv
from random import randint
import MainSimulation_RTS96 as ms



class PRAgent:
	def __init__(self, **kwargs):
		self.memory = []
		



def subtractToZero(originalNumber, subtractionNumber):
	
	if originalNumber > 0:
		if originalNumber - subtractionNumber < 0:
			result = 0 
		else:
			result = originalNumber - subtractionNumber
	elif originalNumber < 0:
		if originalNumber - subtractionNumber > 0:
			result = 0 
		else:
			result = originalNumber - subtractionNumber
	
	else:
		return 0
			
	return result
		
		
def shedAmountIndBus(Case, busNumber, busID):

	cplxPower, cplxCurrent, cplxImpedance = Case.getIndividualBusLoad(int(busNumber), busID, Case.originalCplxPower, Case.originalCplxCurrent, 
		Case.originalCplxImpedance, Case.originalLoad_Numbers, Case.originalLoad_ID_List)

	realPshed = complex(cplxPower).real*(float(Case.shedPercentage)/100)
	imagPshed = complex(cplxPower).imag*(float(Case.shedPercentage)/100)
	realIshed = complex(cplxCurrent).real*(float(Case.shedPercentage)/100)
	imagIshed = complex(cplxCurrent).imag*(float(Case.shedPercentage)/100)
	realZshed = complex(cplxImpedance).real*(float(Case.shedPercentage)/100)
	imagZshed = complex(cplxImpedance).imag*(float(Case.shedPercentage)/100)					
		
	return realPshed, imagPshed, realIshed, imagIshed, realZshed, imagZshed
	

def powerIndBus(Case, busNumber, busID):	
	Case.currentLoadings()
		
	cplxPower, cplxCurrent, cplxImpedance = Case.getIndividualBusLoad(int(busNumber), busID, Case.cplxPower, Case.cplxCurrent, 
			Case.cplxImpedance, Case.Load_Numbers, Case.Load_ID_List)		
	
	
	realPnow = complex(cplxPower).real 
	imagPnow = complex(cplxPower).imag 
	realInow = complex(cplxCurrent).real 
	imagInow = complex(cplxCurrent).imag 
	realZnow = complex(cplxImpedance).real
	imagZnow = complex(cplxImpedance).imag	
	
	return realPnow, imagPnow, realInow, imagInow, realZnow, imagZnow
	

def shedIndividualLoad(Case, busNumber, busID):
	realPshed, imagPshed, realIshed, imagIshed, realZshed, imagZshed = shedAmountIndBus(Case, busNumber, busID)
	realPnow, imagPnow, realInow, imagInow, realZnow, imagZnow = powerIndBus(Case, busNumber, busID)
	realPdif = subtractToZero(realPnow, realPshed)
	imagPdif = subtractToZero(imagPnow, imagPshed)
	realIdif = subtractToZero(realInow, realIshed)
	imagIdif = subtractToZero(imagInow, imagIshed)
	realZdif = subtractToZero(realZnow, realZshed)
	imagZdif = subtractToZero(imagZnow, imagZshed)
	Case.changeBusLoad(int(busNumber), busID, realPdif, imagPdif, realIdif, imagIdif, realZdif, imagZdif)	

def parseDataFile(fileName):
	if 'Network not converged' in open(fileName).read():
		return True 
	
	return False


	
	
def startSimulation(Case):
	Case.loadCase()
	unused, okSol = Case.Solve_Steady()	
	Case.setupDynamicSimulation()
	Case.originalLoadings()
	Case.startDynamicSimulation('output.out',2)	
	Case.contingencyLineTrips()
	Case.runDynamicStep(Case.time+3)
	
	save(self, 'masterSim.sav')
	saveSnap(self, 'masterSim.snp')
	Case.caseName = 'masterSim.sav'
	
def rollout(actionsToEvaluate, Case):
	
	beta = 1
	totalReward = {}
	
	
	flag = True
	for x in range(0, len(actionsToEvaluate)):
		Case.loadCase()
		
		reward = 0
		currentState = 0
		
		if actionsToEvaluate[x] == None:
			Case.currentLoadings()
			while Case.time < 50:
				#print Case.time
				Case.runDynamicStep(Case.time+1)
				if parseDataFile('data.DAT'):
					if Case.time > 10:
						flag = False
					print 'blackout', Case.time, 'NONE'
					break
				
				state, voltages, angles = Case.getState()
				Case.currentLoadings()
				reward += beta**currentState*Case.reward(voltages, Case.cplxPower, Case.cplxCurrent, Case.cplxImpedance, Case.originalCplxPower, 
				Case.originalCplxCurrent, Case.originalCplxImpedance)
				currentState += 1
			#print min(voltages)	
				
		else:
			if type(actionsToEvaluate[x]) != type([]):
				actionsToEvaluate[x] = [actionsToEvaluate[x]]
			
			for y in range(0, len(actionsToEvaluate[x])):
		
				busID = actionsToEvaluate[x][y].split('_')[1]
				busNumber = actionsToEvaluate[x][y].split('_')[2]			
				shedIndividualLoad(Case, busNumber, busID)
			
			Case.currentLoadings()
			while Case.time < 50:
			
				Case.runDynamicStep(Case.time+1)
				if parseDataFile('data.DAT'):
					if Case.time > 10:
						flag = False
					print 'blackout', Case.time
					break
				state, voltages, angles = Case.getState()
				#print min(voltages)
				Case.currentLoadings()
				reward += beta**currentState*Case.reward(voltages, Case.cplxPower, Case.cplxCurrent, Case.cplxImpedance, Case.originalCplxPower, 
				Case.originalCplxCurrent, Case.originalCplxImpedance)
				currentState += 1
			#print min(voltages)
		totalReward[str(actionsToEvaluate[x])] = reward 

	for key in totalReward:
		print key, totalReward[key]
		
		
		
	return flag
def main():
	Case = ms.Network(caseName="RTS96_Static.sav", dynamicsName="Dynamics")
	Case.loadCase()
	Case.originalLoadings()
	#Case.changeInitConds(Case.originalLoad_Numbers, Case.originalLoad_ID_List,  Case.originalCplxPower, Case.originalCplxCurrent, Case.originalCplxImpedance)

	#Case.addLoadSheddingActions(Case.originalLoad_Numbers, Case.originalLoad_ID_List, 10)	
	
	busZones = Case.getBusesInZone()
	
	
	
	#Case.addZoneLoadSheddingActions(busZones, Case.originalLoad_Numbers, Case.originalLoad_ID_List, 10)

	Case.save('temp.sav')
	Case.caseName = 'temp.sav'	
	
	branches = Case.getBranches()
	contingencyList = Case.nMinumsTwoContingencies(branches)
	#contingencyList = [['114_116_1', '116_119_1'], ['116_119_1', '114_116_1'], ['312_323_1', '313_323_1'], ['313_323_1', '312_323_1']]
	#contingencyList = [['114_116_1', '116_119_1']]
	'''
	contingencyList = [['101_103_1', '116_117_1'], ['103_109_1', '115_116_1'], ['103_109_1', '116_117_1'], ['103_109_1', '223_318_1'], 
		['107_108_1', '223_318_1'], ['114_116_1', '303_324_1'], ['114_116_1', '315_324_1'], ['115_116_1', '103_109_1'], ['115_116_1', '121_122_1'], 
		['115_116_1', '121_325_1'], ['115_116_1', '306_310_1'], ['115_121_1', '213_223_1'], ['115_121_2', '213_223_1'], ['116_117_1', '101_103_1'], 
		['116_117_1', '103_109_1'], ['116_117_1', '117_118_1'], ['116_117_1', '117_122_1'], ['116_117_1', '121_122_1'], ['116_117_1', '207_208_1'], 
		['116_117_1', '301_302_1'], ['116_117_1', '303_324_1'], ['116_117_1', '306_310_1'], ['116_117_1', '309_312_1'], ['116_117_1', '311_313_1'], 
		['116_117_1', '315_324_1'], ['116_119_1', '303_324_1'], ['116_119_1', '315_324_1'], ['117_118_1', '116_117_1'], ['117_122_1', '116_117_1'], 
		['121_122_1', '115_116_1'], ['121_122_1', '116_117_1'], ['121_325_1', '115_116_1'], ['203_209_1', '223_318_1'], 
		['207_208_1', '116_117_1'], ['208_210_1', '223_318_1'], ['211_213_1', '223_318_1'], ['211_214_1', '223_318_1'], ['212_223_1', '315_324_1'], 
		['213_223_1', '115_121_1'], ['213_223_1', '115_121_2'], ['213_223_1', '303_324_1'], ['213_223_1', '315_324_1'], ['217_218_1', '223_318_1'], 
		['223_318_1', '103_109_1'], ['223_318_1', '107_108_1'], ['223_318_1', '203_209_1'], ['223_318_1', '208_210_1'], ['223_318_1', '211_213_1'], 
		['223_318_1', '211_214_1'], ['223_318_1', '217_218_1'], ['223_318_1', '301_302_1'], ['223_318_1', '306_310_1'], ['223_318_1', '310_312_1'],
		['223_318_1', '311_313_1'], ['223_318_1', '313_323_1'], ['223_318_1', '316_317_1'], ['301_302_1', '116_117_1'], ['301_302_1', '223_318_1'], 
		['303_324_1', '114_116_1'], ['303_324_1', '116_117_1'], ['303_324_1', '116_119_1'], ['303_324_1', '213_223_1'], ['303_324_1', '306_310_1'], 
		['303_324_1', '310_311_1'], ['303_324_1', '317_322_1'], ['303_324_1', '321_322_1'], ['306_310_1', '115_116_1'], ['306_310_1', '116_117_1'], 
		['306_310_1', '223_318_1'], ['306_310_1', '303_324_1'], ['306_310_1', '315_324_1'], ['309_312_1', '116_117_1'], ['310_311_1', '303_324_1'], 
		['310_311_1', '315_324_1'], ['310_312_1', '223_318_1'], ['311_313_1', '116_117_1'], ['311_313_1', '223_318_1'], ['311_313_1', '312_313_1'], 
		['311_313_1', '312_323_1'], ['312_313_1', '311_313_1'], ['312_323_1', '311_313_1'], ['313_323_1', '223_318_1'], ['314_316_1', '316_319_1'], 
		['315_324_1', '114_116_1'], ['315_324_1', '116_117_1'], ['315_324_1', '116_119_1'], ['315_324_1', '212_223_1'], ['315_324_1', '213_223_1'], 
		['315_324_1', '306_310_1'], ['315_324_1', '310_311_1'], ['315_324_1', '317_322_1'], ['315_324_1', '321_322_1'], ['316_317_1', '223_318_1'], 
		['316_319_1', '314_316_1'], ['317_322_1', '303_324_1'], ['317_322_1', '315_324_1'], ['321_322_1', '303_324_1'], ['321_322_1', '315_324_1']]
	
	
	contingencyList = [['115_121_1', '213_223_1'], ['115_121_2', '213_223_1'], ['211_213_1', '223_318_1'], ['213_223_1', '115_121_1'], 
		['213_223_1', '115_121_2'], ['223_318_1', '211_213_1'], ['311_313_1', '312_313_1'], ['311_313_1', '312_323_1'], 
		['312_313_1', '311_313_1'], ['312_323_1', '311_313_1'], ['315_324_1', '321_322_1'], ['321_322_1', '315_324_1']]
		
	'''	
	contingencyList = [['115_121_1', '213_223_1']]
	print len(contingencyList)
	
	startSimulation(Case)
	#Case.createOverCurrentRelays()
	
	
	
	
	
	'''
	for x in range(0, len(contingencyList)):
		Case.contingency = contingencyList[x]
		print Case.contingency
		rollout(Case.networkActions, Case)
		
		
	'''
	
	
	
	'''
		
	
	blackOuts = []
	for x in range(0, len(contingencyList)):
		Case.contingency = contingencyList[x]
		print Case.contingency
		if not rollout(Case.networkActions, Case):
			blackOuts.append(Case.contingency)
	
	
	with open('contingencies.txt', 'w') as f:
		f.write(str(blackOuts))
	'''
	

if __name__ == "__main__":
    main()	