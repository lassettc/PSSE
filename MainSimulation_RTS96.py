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
import qLearner
import numpy as np
import copy as cp
import csv

_i=psspy.getdefaultint() 
_f=psspy.getdefaultreal() 
_s=psspy.getdefaultchar() 
redirect.psse2py()




def kwargsFunc(kwargs, keyword, default):
	if keyword in kwargs:
		return kwargs[keyword]
	else:
		return default

		
		
class Network:
	def __init__(self, **kwargs):
		self.caseName = kwargsFunc(kwargs, 'caseName', None)
		self.dynamicsName = kwargsFunc(kwargs, 'dynamicsName', None)
		self.networkActions = kwargsFunc(kwargs, 'networkActions', [None])
		self.state = kwargsFunc(kwargs, 'dynamicsName', None)
		self.contingency = kwargsFunc(kwargs, 'contingency', None)
		
		
	def initializeCase(self):
		psspy.psseinit(2000000)
		psspy.case(self.caseName) #Load example case 

		
		
	def incrementalSimulation(self, Time_Trip, endTime, Out_File, busIndexLocation):
		psspy.dynamics_solution_param_2([_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f, 0.01,_f,_f,_f,_f,_f])
		psspy.strt(0,Out_File) #Start our case and specify our output file
		psspy.run(0,0.0,1,1,0) #Run until 0 seconds 
		psspy.run(0,Time_Trip,1,1,0) #Run until 0 seconds 
		
		#psspy.dist_branch_trip(323,325,r"""1""") #Open branch between main grid and microgrid
		psspy.dist_branch_trip(223,318,r"""1""") #Open branch between main grid and microgrid
		psspy.change_channel_out_file(Out_File) #Resume our output file	
		

		ierr, timeStep = psspy.dsrval('DELT')

		count = 0
		time = 0
		count2 = 0
		
		psspy.run(0,5.0,1,1,0)
		while time < endTime:
			count += 1
			count2 += 1
			ierr, time = psspy.dsrval('TIME')
			time = time + timeStep
			ierr, busAngles = psspy.abusreal(-1, 2, 'ANGLED')
		
			
			
			angleDif = busAngles[0][busIndexLocation[223]] - busAngles[0][busIndexLocation[318]]
			angleDif2 = busAngles[0][busIndexLocation[323]] - busAngles[0][busIndexLocation[325]]
				
			if angleDif > 180:
				angleDif -= 180
					
			elif angleDif < -180:
				angleDif += 180 

			if angleDif2 > 180:
				angleDif2 -= 180
					
			elif angleDif2 < -180:
				angleDif2 += 180 
					
			if abs(angleDif) < 0.5:
				print 'mmmmmmmmmm'
				psspy.dist_branch_close(223,318,r"""1""") #Close previously opened branch
				psspy.dist_branch_close(323,325,r"""1""") #Close previously opened branch
				break	
		

			psspy.run(0, time,1,1,0)	
		psspy.run(0, endTime,1,1,0)
		
		
	def checkIfSteadyStable(self):
		with open('data.DAT') as f:
			lines = f.readlines()
		
		for x in range(0, len(lines)):
			if 'Blown up after' in lines[x]:
				return False
			elif 'iteration limit' in lines[x]:
				return False 
		
		return True
			
	def getBusesInZone(self):
		busZoneDict = {}
		ierr, busZones = psspy.abusint(-1, 2, 'ZONE')
		ierr, busNums = psspy.abusint(-1, 2, 'NUMBER')	
		for x in range(0, len(busNums[0])):
			key = busZones[0][x]
			if key <= 10:
				key = 0
			elif key <= 20:
				key = 1 
			elif key <= 30:
				key = 2
			else:
				key = 3 
				
				
				
				
			if key not in busZoneDict:
				busZoneDict[key] = [busNums[0][x]]
			else:
				busZoneDict[key].append(busNums[0][x])
				
		return busZoneDict
		
	def runDynamicSimulation(self, outFile, endTime):
		psspy.strt(0,outFile) #Start our case and specify our output file
		psspy.run(0,0.0,1,1,0) #Run until 0 seconds 
		
		
		ierr, timeStep = psspy.dsrval('DELT')
		print timeStep
		time = 0
		
		psspy.dist_branch_trip(223,318,r"""1""") #Open branch between main grid and microgrid
		while time < endTime:
			ierr, time = psspy.dsrval('TIME')
			time = time + timeStep	
			voltages, angles = getVoltAng()
			print angles[0][0]
			psspy.run(0,time,1,1,0) #Run until 0 seconds 
		
		
		
		
	def getBranches(self):
		branchList = []
		ierr, fromBranches = psspy.abrnint(-1, -1, -1, 4, 1, 'FROMNUMBER')
		ierr, toBranches = psspy.abrnint(-1, -1, -1, 4, 1, 'TONUMBER')
		
		for x in range(0, len(fromBranches[0])):
			fromBus = fromBranches[0][x]
			toBus = toBranches[0][x]
			id = 1 
			branchIdentifier = str(fromBus) + '_' + str(toBus) + '_' + str(id)
			
			while branchIdentifier in branchList:
				id += 1 
				branchIdentifier = str(fromBus) + '_' + str(toBus) + '_' + str(id)
			
			branchList.append(branchIdentifier)
			
		
		return branchList
		
	
	def nMinumsTwoContingencies(self, branches):
		contingencies = []
	
		for x in range(0, len(branches)):
			
			for y in range(0, len(branches)):
				if x != y:
					contingencies.append([branches[x], branches[y]])
					
		return contingencies			
	

	def contingencyLineTrips(self):
		if self.contingency == None:
			print 'No contingency exists..'
			return
	
		for x in range(0, len(self.contingency)):
			fromBusNumber = self.contingency[x].split('_')[0]
			toBusNumber = self.contingency[x].split('_')[1]
			brnchID = self.contingency[x].split('_')[2]
			psspy.dist_branch_trip(int(fromBusNumber),int(toBusNumber),str(brnchID))

		
	
	'''
	def createContingencies(self, branches, nMinusWhat, contingency, contingencyList):
		
		
		if contingency != None:
			#print contingency, contingencyList
			contingencyList.append(contingency)
			
			
		if nMinusWhat == 0:
			print contingencyList
			return contingencyList
		
		nMinusWhat -= 1
		for x in range(0, len(branches)):
			contingencyList = self.createContingencies(branches, nMinusWhat, branches[x], contingencyList)
			
		
		return contingencyList

	
	def createContingencies(self, branches, nMinusWhat, contingency, contingencyList):
		
		
		if nMinusWhat == 0:
			return contingency, contingencyList
			
		nMinusWhat -= 1
		for x in range(0, len(branches)):
			
			contingency.append(branches[x])
			contingency, contingencyList = self.createContingencies(branches, nMinusWhat, contingency, contingencyList)
	
		
		
		print contingency
		if len(contingency) != 0:
			for y in range(0, len(branches)):
				temp = contingency[0:-(len(branches))]
				temp.append(contingency[-(len(branches)-y)])
				contingencyList.append(temp)
			
		
		
		
		return [], contingencyList
		'''
			
		
		
	def changeInitConds(self, Load_Numbs, Load_IDs, cplxPower, cplxCurrent, cplxImpedance):
		
		for x in xrange(0, len(Load_Numbs)):
		
		
			#Present_Power = stringToList(cplxPower[x])
			#Present_Current = stringToList(cplxCurrent[x])
			#Present_Impedance = stringToList(cplxImpedance[x])

			Real_Power = complex(cplxPower[x]).real
			Reactive_Power = complex(cplxPower[x]).imag
			#print Real_Power, Reactive_Power
			
			Dif_Real_Load = Real_Power + Real_Power*random.uniform(-0.2,2)
			Dif_Reactive_Load = Reactive_Power + Reactive_Power*random.uniform(-0.2,2)
			psspy.load_chng_4(Load_Numbs[x],Load_IDs[x],[_i,_i,_i,_i,_i,_i],[Dif_Real_Load, Dif_Reactive_Load,_f,_f,_f,_f])
				
	def save(self, filename):
		psspy.save(filename)

		
	def saveSnap(self, filename):
		psspy.snap(filename)
	
	def loadDynamics(self, filename):
		psspy.rstr(filename)
		
		
	def getVoltAng(self):
		ierr, voltages = psspy.abusreal(-1, 2, 'PU')
		ierr, angles = psspy.abusreal(-1, 2, 'ANGLED')
			
		return voltages, angles
	def Solve_Steady(self):
		Ok_Solution = True
		psspy.fnsl([0,0,0,1,1,0,99,0])
		ierr, rarray = psspy.abusreal(-1, 2, 'PU')
		#print rarray
		if min(rarray[0]) < 0.9 or max(rarray[0]) > 1.1:
			Ok_Solution = False
		
		return rarray[0], Ok_Solution
		
	def Convert_Dynamic(self):
		psspy.fdns([0,0,0,1,1,0,99,0]) #Solve fixed slope decoupled newton raphson	
		psspy.cong(0) #Convert our generators using Zsource (Norton Equiv)
		psspy.conl(0,1,1,[0,0],[ 100.0,00.0,00.0, 100.0]) #Convert our loads (represent active power as 100% constant current type, reactive power as 100% constant impedance type.)
		psspy.conl(0,1,2,[0,0],[ 100.0,00.0,00.0, 100.0]) #Convert our loads
		psspy.conl(0,1,3,[0,0],[ 100.0,00.0,00.0, 100.0]) #Convert our loads
		psspy.ordr(0) #Order network for matrix operations
		psspy.fact() #Factorize admittance matrix
		psspy.tysl(0) #Solution for Switching Studies
	def Out_Put_Channels(self):
		psspy.dyre_new([1,1,1,1],self.dynamicsName,"","","") #Open our .dyr file that gives the information for dynamic responses of the machines
		#psspy.chsb(0,1,[-1,-1,-1,1,4,0]) #Setup Dynamic Simulation channels, Machine voltages in this case
		#psspy.chsb(0,1,[-1,-1,-1,1,14,0]) #Bus voltage and angle
		#psspy.chsb(0,1,[-1,-1,-1,1,12,0]) #Frequency
		#psspy.chsb(0,1,[-1,-1,-1,1,2,0]) #Machine Real power
		#psspy.chsb(0,1,[-1,-1,-1,1,3,0]) #Machine Reactive Power
		#psspy.chsb(0,1,[-1,-1,-1,1,1,0]) #Machine Angle
		#psspy.chsb(0,1,[-1,-1,-1,1,7,0]) #Machine Speed
		#psspy.chsb(0,1,[-1,-1,-1,1,14,0])
		
	def PSSE_Arrays2_List(self, In_Array):
		Temp_1 = ''.join(str(e) for e in In_Array)
		Temp_2 = Temp_1.replace("[", "")
		Temp_2 = Temp_2.replace("]", "")
		In_String = Temp_2.split(', ')
		Out_List = list()
		for x in range(0, len(In_String)):	
			Out_List.append(In_String[x])
		return Out_List

	def Load_Duplicate_Fix(self, In_List):
		count = 1
		List = In_List
		List2 = list()
		for x in range(0,len(In_List)):
			Count_Name = str(count)
			In_List[x] = List[x] + '_' + Count_Name
			List2.append(Count_Name)
			if x < len(In_List) - 1:
				if List[x+1] in In_List[x]:
					count = count + 1
				else:
					count = 1
		return In_List, List2
		
	def Lists_2_Dicts(self, Array1, Array2):
		Dictionary = dict(zip(Array1, Array2))
		return Dictionary
		
	def Return_Load_Info(self):	
		ierr, Complex_Power = psspy.aloadcplx(-1, 4, 'MVANOM') #Obtain Complex Power of Loads
		ierr, Complex_Current = psspy.aloadcplx(-1, 4, 'ILNOM') #Obtain Complex Currents of Loads
		ierr, Complex_Impedance = psspy.aloadcplx(-1, 4, 'YLNOM') #Obtain Complex Impedances of Loads
		ierr, Load_Numbers = psspy.aloadint(-1, 4, 'NUMBER') #Obtain Load Numbers
		ierr, Load_Count = psspy.aloadcount(-1, 4) #Obtain Count of Loads

		return Load_Count, Load_Numbers, Complex_Power, Complex_Current, Complex_Impedance

	def Update_Dict_Vals(self, Dict2_Update):
		return

	def loadInfo(self):	
		ierr, cplxPower = psspy.aloadcplx(-1, 2, 'MVANOM') #Obtain Complex Power of Loads
		ierr, cplxCurrent = psspy.aloadcplx(-1, 2, 'ILNOM') #Obtain Complex Currents of Loads
		ierr, cplxImpedance = psspy.aloadcplx(-1, 2, 'YLNOM') #Obtain Complex Impedances of Loads
		ierr, Load_Numbers = psspy.aloadint(-1, 2, 'NUMBER') #Obtain Load Numbers
		ierr, Load_Count = psspy.aloadcount(-1, 2) #Obtain Count of Loads

		return Load_Count, Load_Numbers, cplxPower, cplxCurrent, cplxImpedance


		
	def ZIP_Loads(self):
		Load_Count, Load_Numbers, cplxPower, cplxCurrent, cplxImpedance = self.loadInfo() 
		
		Load_Numbers2 = self.PSSE_Arrays2_List(Load_Numbers)
		
		loadNumbsAndID, Load_ID_List = self.Load_Duplicate_Fix(Load_Numbers2)

		cplxPower = self.PSSE_Arrays2_List(cplxPower)
		cplxCurrent = self.PSSE_Arrays2_List(cplxCurrent)
		cplxImpedance = self.PSSE_Arrays2_List(cplxImpedance)

		return cplxPower, cplxCurrent, cplxImpedance, Load_Numbers, Load_ID_List


	def ZIP_Loadings2_Dicts(self, Complex_Power, Complex_Current, Complex_Impedance):
		Complex_Power_Dict = Lists_2_Dicts(Load_Numbers, Complex_Power)
		Complex_Current_Dict = Lists_2_Dicts(Load_Numbers, Complex_Current)
		Complex_Impedance_Dict = Lists_2_Dicts(Load_Numbers, Complex_Impedance)
		
		return Complex_Power_Dict, Complex_Current_Dict, Complex_Impedance_Dict

	
	def addLoadSheddingActions(self, busNumbers, Load_ID_List, shedAmount):
		self.shedPercentage = shedAmount
		for x in range(0, len(busNumbers)):
			self.networkActions.append('loadShed_' + str(Load_ID_List[x]) + '_' + str(busNumbers[x]))
			
	
	
	
	def parseCosmicPS(self, fileName):
		with open(fileName, 'rb') as f:
			reader = csv.reader(f)
			data = list(reader)		
	
		return data
	
	def getBranchRateFromData(self, data):
		branchRateDict = {}
		#print data[1]
		for x in range(0, len(data)):
			count = 1
			newKey = str(data[x][0]) + '_' + str(data[x][1]) + '_' + str(count)
			while newKey in branchRateDict:
				count += 1
				newKey = str(data[x][0]) + '_' + str(data[x][1]) + '_' + str(count)
				
			branchRateDict[newKey] = data[x][6]
			#print branchRateDict[newKey], newKey
		
		return branchRateDict
		
	def createOverCurrentRelays(self):
		data = self.parseCosmicPS('rts96branchInfo.csv')
		relayTemplate = "101, 'TIOCR1', 102, '1', 1, 1, , , 101, 102, '1', , , , , , , 3, 5, 1.0, 0.2, 55, 0.15, 55, 0.1, 55, 0.05, 2, 0.0, 0.0, 0.1"	
		relayTemplateList = relayTemplate.split(',')
		
		branchRateDict = self.getBranchRateFromData(data)
		
		
		with open("dynamicsTemp.dyr", "a") as myfile:
			for key in branchRateDict:
				toBus = key.split('_')[0]
				fromBus = key.split('_')[1]
				id = key.split('_')[2]
				rate = branchRateDict[key]
				temp = cp.copy(relayTemplateList)
				temp[0], temp[2], temp[8], temp[9] = toBus, fromBus, toBus, fromBus
				temp[3], temp[10] = str(id), str(id)
				temp[17] = str(float(rate)/150)
				temp[19] = str(1.0)
				
				temp = ','.join(temp)
				myfile.write(str(temp))
				myfile.write(str(' /'))
				myfile.write("\n")
	def addZoneLoadSheddingActions(self, busZones, busNumbers, Load_ID_List, shedAmount):
		self.shedPercentage = shedAmount
		for key in busZones:
			tempList = []
			for x in range(0, len(busNumbers)):
				if busNumbers[x] in busZones[key]:
					tempList.append('loadShed_' + str(Load_ID_List[x]) + '_' + str(busNumbers[x]))
				
			self.networkActions.append(cp.copy(tempList))
	
	
	def getBranchFlow(self, type):
		unused, rarray = psspy.aflowreal(-1, 1, -1, 2, type)
		unused, rarray2 = psspy.aflowint(-1, 1, -1, 2, 'FROMNUMBER')
		unused, rarray3 = psspy.aflowint(-1, 1, -1, 2, 'TONUMBER')
		unused, rarray4 = psspy.aflowreal(-1, 1, -1, 2, 'PCTRATEA')
		unused, rarray5 = psspy.aflowreal(-1, 1, -1, 2, 'RATEA')
		'''
		print rarray4[0][32]
		print rarray5[0][32]		
		print rarray2[0][32]
		print rarray3[0][32]
		print rarray[0][32]
		print (rarray4[0][32]/100)*(rarray5[0][32]/100)
		'''
		return rarray[0]
	
	
	def getState(self):
		voltages, angles = self.getVoltAng()
		totalMeasurements = voltages[0] + angles[0]
		state = totalMeasurements
		
		return voltages[0], voltages[0], angles[0]
		
	def runDynamicSim(self):
		self.initializeCase()
		psspy.lines_per_page_one_device(1,10000)   
		psspy.progress_output(2,'data.DAT',[0,0])
		cplxPower, cplxCurrent, cplxImpedance, Load_Numbers, Load_ID_List = self.ZIP_Loads()
		self.addLoadSheddingActions(Load_Numbers[0], Load_ID_List)	
			
			
		ierr, busNumbs = psspy.abusint(-1, 2, 'NUMBER')
		ierr, iarray2 = psspy.abusreal(-1, 2, 'ANGLED')
			
		self.Convert_Dynamic()
		self.Out_Put_Channels()
		self.getState()
		
		#runDynamicSimulation('outFile.out', 25)

	def originalLoadings(self):
		cplxPower, cplxCurrent, cplxImpedance, Load_Numbers, Load_ID_List = self.ZIP_Loads()
	
		self.originalCplxPower = cplxPower
		self.originalCplxCurrent = cplxCurrent
		self.originalCplxImpedance = cplxImpedance
		self.originalLoad_Numbers = Load_Numbers[0]
		self.originalLoad_ID_List = Load_ID_List


	def currentLoadings(self):
		cplxPower, cplxCurrent, cplxImpedance, Load_Numbers, Load_ID_List = self.ZIP_Loads()
	
		self.cplxPower = cplxPower
		self.cplxCurrent = cplxCurrent
		self.cplxImpedance = cplxImpedance
		self.Load_Numbers = Load_Numbers[0]
		self.Load_ID_List = Load_ID_List

		busNumbers = psspy.abusint(-1, 2, 'NUMBER')
		self.busNumbers = busNumbers[1][0]
		
		
		
	def getIndividualBusLoad(self, busNumber, busID, cplxPower, cplxCurrent, cplxImpedance, loadNumbers, loadIDs):
		for y in range(0, len(loadNumbers)):
			if busNumber == loadNumbers[y] and busID == loadIDs[y]:
				return cplxPower[y], cplxCurrent[y], cplxImpedance[y]
		
		return 0, 0, 0
	
	


	def loadCase(self):
		self.initializeCase()
		
	
	def setupDynamicSimulation(self):
		
		psspy.lines_per_page_one_device(1,10000)   
		psspy.progress_output(2,'data.DAT',[0,0])
		#cplxPower, cplxCurrent, cplxImpedance, Load_Numbers, Load_ID_List = self.ZIP_Loads()
		#self.addLoadSheddingActions(Load_Numbers[0], Load_ID_List, 50)	
		self.Convert_Dynamic()
		self.Out_Put_Channels()		
		
	

	def startDynamicSimulation(self,outFile,endTime):
		self.outfile = outFile
		psspy.strt(0,outFile) #Start our case and specify our output file
		psspy.run(0,0.0,1,1,0) #Run until 0 seconds 	
		psspy.run(0, endTime,1,1,0)
		ierr, timeStep = psspy.dsrval('DELT')
		self.timeStep = timeStep
		ierr, time = psspy.dsrval('TIME')
		self.time = time
		
		
	def runDynamicStep(self, endTime):
		psspy.run(0, endTime,1,1,0)
		ierr, time = psspy.dsrval('TIME')
		self.time = time
	
	def busNumberIndexDict(self, inputBusNumbs):
		dict = {}
		for x in xrange(len(inputBusNumbs)):
			dict[inputBusNumbs[x]] = x 

		return dict

	def changeBusLoad(self, busNumber, busID, realP, imagP, realI, imagI, realZ, imagZ):
		
		
		psspy.load_chng_4(busNumber,busID,[_i,_i,_i,_i,_i,_i],[realP, imagP, realI, imagI, realZ, imagZ])
		
	
	
	def getComplexRealLoadRewardRatio(self, origLoad, currentLoad, count, reward):
		try:
			#print float(complex(currentLoad).real)/complex(origLoad).real
			reward += float(complex(currentLoad).real)/complex(origLoad).real
			if float(complex(currentLoad).real)/complex(origLoad).real < 1:
				pass
				#print "K"
				
				
			count += 1
			
		except ZeroDivisionError:
			pass
		return reward, count			

	def getComplexImagLoadRewardRatio(self, origLoad, currentLoad, count, reward):
		try:
			
			#print float(complex(currentLoad).imag)/complex(origLoad).imag
			reward += float(complex(currentLoad).imag)/complex(origLoad).imag
			
			if float(complex(currentLoad).imag)/complex(origLoad).imag < 1:
				pass
				#print "K"
			count += 1
	
	
			
			
		except ZeroDivisionError:
			pass
		return reward, count				




	def reward(self, state, cplxPower, cplxCurrent, cplxImpedance, originalCplxPower, originalCplxCurrent, originalCplxImpedance):
		reward = 0
		totalReward = 0
		count = 0
		for y in range(0, len(cplxPower)):
			#print originalCplxCurrent[y], cplxCurrent[y], self.Load_Numbers[y]
			reward, count = self.getComplexRealLoadRewardRatio(originalCplxPower[y], cplxPower[y], count, reward)
			reward, count = self.getComplexImagLoadRewardRatio(originalCplxPower[y], cplxPower[y], count, reward)
			reward, count = self.getComplexRealLoadRewardRatio(originalCplxCurrent[y], cplxCurrent[y], count, reward)
			reward, count = self.getComplexImagLoadRewardRatio(originalCplxCurrent[y], cplxCurrent[y], count, reward)
			reward, count = self.getComplexRealLoadRewardRatio(originalCplxImpedance[y], cplxImpedance[y], count, reward)
			reward, count = self.getComplexImagLoadRewardRatio(originalCplxImpedance[y], cplxImpedance[y], count, reward)			
			#print reward, count
			
		
	
		totalReward = float(reward)/count
	
	
			
		return totalReward	











		
	def rewardTypeOne(self, state, cplxPower, cplxCurrent, cplxImpedance, originalCplxPower, originalCplxCurrent, originalCplxImpedance):
		reward = 0
		totalReward = 0
		count = 0
		for y in range(0, len(cplxPower)):
			#print originalCplxCurrent[y], cplxCurrent[y], self.Load_Numbers[y]
			reward, count = self.getComplexRealLoadRewardRatio(originalCplxPower[y], cplxPower[y], count, reward)
			reward, count = self.getComplexImagLoadRewardRatio(originalCplxPower[y], cplxPower[y], count, reward)
			reward, count = self.getComplexRealLoadRewardRatio(originalCplxCurrent[y], cplxCurrent[y], count, reward)
			reward, count = self.getComplexImagLoadRewardRatio(originalCplxCurrent[y], cplxCurrent[y], count, reward)
			reward, count = self.getComplexRealLoadRewardRatio(originalCplxImpedance[y], cplxImpedance[y], count, reward)
			reward, count = self.getComplexImagLoadRewardRatio(originalCplxImpedance[y], cplxImpedance[y], count, reward)			
			
			#print reward, count
			
			
	
		totalReward = float(reward)/count
		
		
		for x in range(0, len(state)):
			if state[x] < 0.97:
				totalReward -= 100*(0.97-state[x])
			
			
		return totalReward	

	def rewardTypeTwo(self, state, cplxPower, cplxCurrent, cplxImpedance, originalCplxPower, originalCplxCurrent, originalCplxImpedance):
		totalReward = 0
		reward = 0
		count = 0
		loadIndex = 0
		loadNumbs = self.Load_Numbers
		loadNumbs.append(None)
		for x in range(0, len(state)):
			#print x, loadIndex, len(state), len(loadNumbs)
			if self.busNumbers[x] == loadNumbs[loadIndex]:
				#print self.busNumbers[x], loadNumbs[loadIndex]
				if state[x] > 0.97:
					reward, count = self.getComplexRealLoadRewardRatio(originalCplxPower[loadIndex], cplxPower[loadIndex], count, reward)
					reward, count = self.getComplexImagLoadRewardRatio(originalCplxPower[loadIndex], cplxPower[loadIndex], count, reward)
					reward, count = self.getComplexRealLoadRewardRatio(originalCplxCurrent[loadIndex], cplxCurrent[loadIndex], count, reward)
					reward, count = self.getComplexImagLoadRewardRatio(originalCplxCurrent[loadIndex], cplxCurrent[loadIndex], count, reward)
					reward, count = self.getComplexRealLoadRewardRatio(originalCplxImpedance[loadIndex], cplxImpedance[loadIndex], count, reward)
					reward, count = self.getComplexImagLoadRewardRatio(originalCplxImpedance[loadIndex], cplxImpedance[loadIndex], count, reward)	
					
				else:
					reward -= 100000*(0.97-state[x])
				
				loadIndex += 1
				
		totalReward = float(reward)/count
		
		
		return totalReward
def main():
	pass

if __name__ == "__main__": 
	main()	



