# File:"C:\Program Files (x86)\PTI\PSSEXplore33\EXAMPLE\My_py.py", generated on TUE, AUG 11 2015   8:29, release 33.05.02
from __future__ import division
from collections import defaultdict
import os,sys
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN") #Give the path to PSSBIN to imoport psspy
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;" #Tell PSSE where "itself" is
                      + os.environ['PATH'])
					  
import pssarrays				  
import psspy
import redirect
import dyntools
import pssplot
import random
import math
#import matplotlib
_i=psspy.getdefaultint() 
_f=psspy.getdefaultreal() 
_s=psspy.getdefaultchar() 
redirect.psse2py()


def initializeCase(inFile):
	psspy.psseinit(2000000)
	psspy.case(inFile) #Load example case savnw.sav

def returnNewIslandsInformation():
	ierr, inServiceBuses = psspy.abusint(-1, 1, 'NUMBER')
	ierr, allBuses = psspy.abusint(-1, 2, 'NUMBER')
	outOfServiceBuses = len(allBuses[0]) - len(inServiceBuses[0])
	busIslands = []
	ierr, buses = psspy.tree(1, 1)
	while buses != 0:
		busIslands.append(buses)
		ierr, buses = psspy.tree(2, 1)	
	
	return busIslands, outOfServiceBuses

def loadBranchAndMachineInfo(caseFile):
	initializeCase(caseFile)
	ierr, totalMachines = psspy.amachcount(-1, 4)
	ierr, inserviceMachines = psspy.amachcount(-1, 1)
	ierr, realCurrentInService = psspy.aloadreal(-1, 1, 'ILNOM')
	return totalMachines, inserviceMachines, realCurrentInService
	
	
def analyzeGoodOrBadCases(saveTo):
	newPath = os.path.join(saveTo, 'Unstable')
	
	if not os.path.exists(newPath):
		os.makedirs(newPath)
	countGoodCases = 0
	goodCaseInserviceMachinesBeforeRecon = []
	goodCaseInserviceMachinesAfterRecon = []
	goodCaseMachineAverage = []
	
	badCaseInserviceMachinesBeforeRecon = []
	badCaseInserviceMachinesAfterRecon = []
	badCaseMachineAverage = []
	
	goodCaseLoadBeforeRecon = []
	goodCaseLoadAfterRecon = []
	goodCaseLoadAverage = [] 
	badCaseLoadAverage = []
	
	
	for x in range(0, 50):
		if not os.path.exists(os.path.join(saveTo, 'polandIslandingTestControl_' + str(x) + 'BeforeRecon.sav')):
			return countGoodCases
		beforeReconFileName = 'polandIslandingTestControl_' + str(x) + 'BeforeRecon.sav'	
		beforeReconFile = os.path.join(saveTo, beforeReconFileName)	
		iFile3Name = 'polandIslandingTestControl_' + str(x) + '.out'
		inFile3 = os.path.join(saveTo, iFile3Name)
		initializeCase(beforeReconFile)
		psspy.lines_per_page_one_device(1,10000)   
		psspy.progress_output(2,os.path.join(saveTo, 'OutputExtract'),[0,0])
		psspy.bus_chng_3(18,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
		busIslandsBeforeRecon, outOfServiceBeforeRecon = returnNewIslandsInformation()
		
		afterReconFileName = 'polandIslandingTestControl_' + str(x) + '.sav'
		afterReconFile = os.path.join(saveTo, afterReconFileName)
		initializeCase(afterReconFile)
		psspy.bus_chng_3(18,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
		busIslandsAfterRecon, outOfServiceAfterRecon = returnNewIslandsInformation()
		#print 'CASE: ', str(x), ' Information'
		#print 'Before Reconnection; ', 'Buses in Islands: ', busIslandsBeforeRecon, 'Total buses count: ', sum(busIslandsBeforeRecon), 'Out of service buses: ', outOfServiceBeforeRecon
		#print 'After Reconnection; ', 'Buses in Islands: ', busIslandsAfterRecon, 'Total buses count: ', sum(busIslandsAfterRecon), 'Out of service buses: ', outOfServiceAfterRecon
	
		
		if max(busIslandsAfterRecon) > 2330 and outOfServiceAfterRecon < 53:
			
			countGoodCases += 1
			totalMachinesBeforeRecon, inserviceMachinesBeforeRecon, realCurrentInServiceBeforeRecon = loadBranchAndMachineInfo(beforeReconFile)
			totalMachinesAfterRecon, inserviceMachinesAfterRecon, realCurrentInServiceAfterRecon = loadBranchAndMachineInfo(afterReconFile)
			print 'Good case:'
			print max(busIslandsAfterRecon), outOfServiceAfterRecon
			print totalMachinesBeforeRecon, inserviceMachinesBeforeRecon, sum(realCurrentInServiceBeforeRecon[0])
			print totalMachinesAfterRecon, inserviceMachinesAfterRecon, sum(realCurrentInServiceAfterRecon[0])
		
			goodCaseLoadAverage.append(sum(realCurrentInServiceAfterRecon[0])/sum(realCurrentInServiceBeforeRecon[0]))
			goodCaseMachineAverage.append(float(inserviceMachinesAfterRecon)/inserviceMachinesBeforeRecon)
		else:
			totalMachinesBeforeRecon, inserviceMachinesBeforeRecon, realCurrentInServiceBeforeRecon = loadBranchAndMachineInfo(beforeReconFile)
			totalMachinesAfterRecon, inserviceMachinesAfterRecon, realCurrentInServiceAfterRecon = loadBranchAndMachineInfo(afterReconFile)
			print 'Bad case:'
			print max(busIslandsAfterRecon), outOfServiceAfterRecon
			print totalMachinesBeforeRecon, inserviceMachinesBeforeRecon, sum(realCurrentInServiceBeforeRecon[0])
			print totalMachinesAfterRecon, inserviceMachinesAfterRecon, sum(realCurrentInServiceAfterRecon[0])
			
			print beforeReconFile
			os.rename(beforeReconFile, os.path.join(newPath, beforeReconFileName))
			os.rename(afterReconFile, os.path.join(newPath, afterReconFileName))
			os.rename(inFile3, os.path.join(newPath, iFile3Name))
			badCaseLoadAverage.append(sum(realCurrentInServiceAfterRecon[0])/sum(realCurrentInServiceBeforeRecon[0]))
			badCaseMachineAverage.append(float(inserviceMachinesAfterRecon)/inserviceMachinesBeforeRecon)
	print countGoodCases
	
	try:
		print 'good case load average min: ', min(goodCaseLoadAverage)
	except ValueError:
		pass
		
	try:
		print 'bad case load average min: ', min(badCaseLoadAverage)
	except ValueError:
		pass
		
	try:
		print 'good case machine average min: ', min(goodCaseMachineAverage)
	except ValueError:
		pass
		
	try:
		print 'bad case mahine average min: ', min(badCaseMachineAverage)
	except ValueError:
		pass

	
def main():
	newPath = os.getcwd() + '/UnstableCases/'
	analyzeGoodOrBadCases(newPath)
	pass
	'''	
		else:
			os.rename(inFile, newPath + inFile)
			os.rename(inFile2, newPath + inFile2)
			os.rename(inFile3, newPath + inFile3)
			print busIslandsAfterRecon
	print countGoodCases
	'''
	'''
	
	for x in range(0, 300):
		inFile = 'StableCasespolandIslandingTestControl_' + str(x) + '.sav'
		if os.path.exists(inFile):
			os.rename(inFile, newPath + 'polandIslandingTestControl_' + str(x) + '.sav')
	'''	
if __name__ == "__main__": 
	main()	