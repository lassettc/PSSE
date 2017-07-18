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

dynamic_case_file = 'staticCasePoland.sav'
inFile = 'polandCase.sav'
dynamicFile = 'PolandDynamicsProtection.dyr'
def initializeCase(inFile):
	psspy.psseinit(2000000)
	psspy.case(inFile) #Load example case savnw.sav

def changeInitConds(busInZone, Load_Numbs, Load_IDs, cplxPower, cplxCurrent, cplImpedance):
	
	for x in xrange(0, len(Load_Numbs)):
		if Load_Numbs[x] in busInZone:
			Present_Power = stringToList(cplxPower[x])
			Present_Current = stringToList(cplxCurrent[x])
			Present_Impedance = stringToList(cplImpedance[x])

			Real_Power = float(Present_Power[0])
			Reactive_Power = float(Present_Power[1])
			
			Dif_Real_Load = Real_Power + Real_Power*random.uniform(-0.1,0.1)
			Dif_Reactive_Load = Reactive_Power + Reactive_Power*random.uniform(-0.1,0.1)
			psspy.load_chng_4(Load_Numbs[x],Load_IDs[x],[_i,_i,_i,_i,_i,_i],[Dif_Real_Load, Dif_Reactive_Load,_f,_f,_f,_f])
			


		
def Solve_Steady():
	Ok_Solution = 1
	psspy.fnsl([0,0,0,1,1,0,99,0])
	ierr, rarray = psspy.abusreal(-1, 2, 'PU')
	#print rarray
	print min(rarray[0])
	print max(rarray[0])
	if min(rarray[0]) < 0.8 or max(rarray[0]) > 1.2:
		Ok_Solution = 0
	
	return rarray, Ok_Solution
	
def Convert_Dynamic():
	psspy.fdns([0,0,0,1,1,0,99,0]) #Solve fixed slope decoupled newton raphson	
	psspy.cong(0) #Convert our generators using Zsource (Norton Equiv)
	psspy.conl(0,1,1,[0,0],[ 100.0,00.0,00.0, 100.0]) #Convert our loads (represent active power as 100% constant current type, reactive power as 100% constant impedance type.)
	psspy.conl(0,1,2,[0,0],[ 100.0,00.0,00.0, 100.0]) #Convert our loads
	psspy.conl(0,1,3,[0,0],[ 100.0,00.0,00.0, 100.0]) #Convert our loads
	psspy.ordr(0) #Order network for matrix operations
	psspy.fact() #Factorize admittance matrix
	psspy.tysl(0) #Solution for Switching Studies
	#psspy.save(savFile2)
def outPutChannels():
	psspy.dyre_new([1,1,1,1],dynamicFile,"","","") #Open our .dyr file that gives the information for dynamic responses of the machines
	
	psspy.chsb(0,1,[-1,-1,-1,1,4,0]) #Setup Dynamic Simulation channels, Machine voltages in this case
	#psspy.chsb(0,1,[-1,-1,-1,1,14,0]) #Bus voltage and angle
	psspy.chsb(0,1,[-1,-1,-1,1,12,0]) #Frequency
	psspy.chsb(0,1,[-1,-1,-1,1,2,0]) #Machine Real power
	psspy.chsb(0,1,[-1,-1,-1,1,3,0]) #Machine Reactive Power
	#psspy.chsb(0,1,[-1,-1,-1,1,1,0]) #Machine Angle
	psspy.chsb(0,1,[-1,-1,-1,1,7,0]) #Machine Speed
	psspy.chsb(0,1,[-1,-1,-1,1,14,0])
	psspy.branch_p_channel([-1,-1,-1,125,126],r"""1""","")
	
	

def flatStart(outFile, endTime):
	'''Function that runs a flat start simulation to ensure everything is 
	operating to satisfaction before any in depth sims.
	'''
	psspy.strt(0,outFile) #Start our case and specify our output file
	psspy.run(0,0.0,1,1,0) #Run until 0 seconds 
	psspy.run(0, endTime,1,1,0) #Run until end time

	
def checkFileForPhrase(phrase, fileName):
	if phrase in open(fileName).read():
		return True
	else:
		return False
	
	
def Islanding(busInZone, Time_Trip, timeLoadChange, Time_Reconnect, End_Time, Out_File, toBus, fromBus, savFile2, outputDataFile, saveTo):
	zone = 5
	masterConvergenceFile = os.path.join(saveTo, 'MasterConvergenceFile.txt')
	psspy.strt(0,Out_File) #Start our case and specify our output file
	psspy.run(0,0.0,1,1,0) #Run until 0 seconds 
	#psspy.change_channel_out_file(Out_File) #Resume output file
	psspy.run(0, Time_Trip,1,1,0) #Run until 50 seconds
	if 'PICKUP' in open(outputDataFile).read() or 'Network not converged' in open(outputDataFile).read():
		return 0
		
	firstCurrentTime = Time_Trip
	print toBus, fromBus, len(toBus)
	for x in range(0, len(toBus)):
		firstCurrentTime += float(1)/60
		psspy.dist_branch_trip(toBus[x],fromBus[x],r"""1""")

	
	if 'Network not converged' in open(outputDataFile).read():
		return 0
			

	psspy.run(0, firstCurrentTime + 5,1,1,0)
	if 'Network not converged' in open(outputDataFile).read():
		return 0
	
	psspy.run(0, Time_Reconnect,1,1,0)
	currentTime = Time_Reconnect
	psspy.save(savFile2)
	for y in range(0, len(toBus)):
		currentTime += float(1)/60
		psspy.dist_branch_close(toBus[y],fromBus[y],r"""1""") #Close previously opened branch
		psspy.run(0, currentTime,1,1,0)
		
	if 'Network not converged' in open(outputDataFile).read():
		with open(masterConvergenceFile, 'a') as myfile:
			myfile.write(Out_File)
			myfile.write(', ')
			myfile.write('0')
			myfile.write('\n')
			return 1
			
	networkConvergenceCheck = currentTime + 3
	psspy.run(0, networkConvergenceCheck,1,1,0)	
	if 'Network not converged' in open(outputDataFile).read():
		with open(masterConvergenceFile, 'a') as myfile:
			myfile.write(Out_File)
			myfile.write(', ')
			myfile.write('0')
			myfile.write('\n')
			return	1
			
	networkConvergenceCheck2 = networkConvergenceCheck + 3		
	psspy.run(0, networkConvergenceCheck2,1,1,0)	
	if 'Network not converged' in open(outputDataFile).read():
		with open(masterConvergenceFile, 'a') as myfile:
			myfile.write(Out_File)
			myfile.write(', ')
			myfile.write('0')
			myfile.write('\n')
			return	1		
			
	psspy.run(0, End_Time,1,1,0)
	if 'Network not converged' in open(outputDataFile).read():
		with open(masterConvergenceFile, 'a') as myfile:
			myfile.write(Out_File)
			myfile.write(', ')
			myfile.write('0')
			myfile.write('\n')
			return 1
	zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss = zoneTotalsPsseFunc()
	microGridRealPower, microGridReactivePower, microGridRealLoad, microGridReactiveLoad, microGridRealLoss, microGridReactiveLoss, mainGridRealPower, mainGridReactivePower, mainGridRealLoad, mainGridReactiveLoad, mainGridRealLoss, mainGridReactiveLoss = inOutMicroGridTotals(zone, zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss)
		
	psspy.run(0, End_Time,1,1,0) #Run case until 125 seconds
	
	return 1
def Load_Change(Time_Trip, Time_Reconnect, End_Time, Out_File):
	psspy.strt(0,Out_File) #Start our case and specify our output file
	psspy.run(0,0.0,1,1,0) #Run until 0 seconds 
	psspy.run(0, Time_Trip,1,1,0) #Run until 5 seconds
	psspy.load_chng_4(3007,r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f, _f,_f,_f,_f])
	psspy.load_chng_4(3008,r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f, _f,_f,_f,_f])
	psspy.load_chng_4(3005,r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f, _f,_f,_f,_f])
	psspy.change_channel_out_file(Out_File) #Resume our output file
	psspy.run(0, Time_Reconnect,1,1,0) #Run case until 15 seconds
	psspy.change_channel_out_file(Out_File) #Resume output file
	psspy.run(0, End_Time,1,1,0) #Run case until 125 seconds
	#psspy.snap([189,67,10,0,0],r"""C:\Program Files (x86)\PTI\PSSEXplore33\EXAMPLE\Dynamic_Models_NoWarnings_new.snp""") #Take a snapshot of the system after completion of simulation


	#loading_current = 1122.449
	#loading_impedance = -520.6165
	Time = 10 - 1/120
	psspy.strt(0,Out_File) #Start our case and specify our output file
	psspy.run(0, Time,1,1,0) #Run until 5 seconds
	
	
	loading_current153 = 198.9457 
	loading_impedance153 = -98.9485

	loading_current154 = 620.5281
	loading_impedance154 = -481.319 

	loading_current154_2 =  413.6854 
	loading_impedance154_2 = -374.359 

	loading_current203 =  303.3805
	loading_impedance203 = -153.4

	loading_current205 = 1122.449 
	loading_impedance205 = -520.6165 

	loading_current3005 = 100.4083 
	loading_impedance3005 = -50.4092 

	loading_current3007 = 205.6508
	loading_impedance3007 = -79.298 

	loading_current3008 = 205.2303
	loading_impedance3008 = -78.974 

	loading_current3019 = 100 
	loading_impedance3019 = -50 
	
	Current_153Max_Swing = 3
	Corr_Ratio = loading_current153/Current_153Max_Swing
	
	Current_154Max_Swing = loading_current154/Corr_Ratio
	Current_154_2Max_Swing = loading_current154_2/Corr_Ratio
	Current_203Max_Swing = loading_current203/Corr_Ratio
	Current_205Max_Swing = loading_current205/Corr_Ratio
	Current_3005Max_Swing = loading_current3005/Corr_Ratio
	Current_3007Max_Swing = loading_current3007/Corr_Ratio
	Current_3008Max_Swing = loading_current3008/Corr_Ratio
	Current_3019Max_Swing = loading_current3019/Corr_Ratio
	
	Impedance_153Max_Swing = loading_impedance153/Corr_Ratio
	Impedance_154Max_Swing = loading_impedance154/Corr_Ratio
	Impedance_154_2Max_Swing = loading_impedance154_2/Corr_Ratio
	Impedance_203Max_Swing = loading_impedance203/Corr_Ratio
	Impedance_205Max_Swing = loading_impedance205/Corr_Ratio
	Impedance_3005Max_Swing = loading_impedance3005/Corr_Ratio
	Impedance_3007Max_Swing = loading_impedance3007/Corr_Ratio
	Impedance_3008Max_Swing = loading_impedance3008/Corr_Ratio
	Impedance_3019Max_Swing = loading_impedance3019/Corr_Ratio
	
	ierr, rval2 = psspy.dsrval('DELT')
	ierr, Time = psspy.dsrval('TIME')
	Dummy_Time = Time + rval2
	for x in range(0,36000):
		ierr, Time = psspy.dsrval('TIME')
		Time = Time + rval2
		Converted_Time = int(Start_Trip*120)
		#loading_current205 = (1122.449*(1 + 0.05*math.sin(Time-5)) + math.sin(Time-5))
		#loading_impedance205 = (-520.6165 + math.sin(Time-5))
		
		Change_loading_current153 = loading_current153 + Current_153Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance153 = loading_impedance153 + Impedance_153Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current154 = loading_current154 + Current_154Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance154 = loading_impedance154 + Impedance_154Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current154_2 =  loading_current154_2 + Current_154_2Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance154_2 = loading_impedance154_2 + Impedance_154_2Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current203 =  loading_current203 + Current_203Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance203 = loading_impedance203 + Impedance_203Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current205 = loading_current205 + Current_205Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance205 = loading_impedance205 + Impedance_205Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current3005 = loading_current3005 + Current_3005Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance3005 = loading_impedance3005 + Impedance_3005Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current3007 = loading_current3007 + Current_3007Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance3007 = loading_impedance3007 + Impedance_3007Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current3008 = loading_current3008 + Current_3008Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance3008 = loading_impedance3008 + Impedance_3008Max_Swing*math.sin(Time-Dummy_Time)

		Change_loading_current3019 = loading_current3019 + Current_3019Max_Swing*math.sin(Time-Dummy_Time)
		Change_loading_impedance3019 = loading_impedance3019 + Impedance_3019Max_Swing*math.sin(Time-Dummy_Time)
		
		
		
		psspy.load_chng_4(153,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current153,_f,_f,_f])
		psspy.load_chng_4(153,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance153])
		
		psspy.load_chng_4(154,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current154,_f,_f,_f])
		psspy.load_chng_4(154,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance154])
		
		psspy.load_chng_4(154,r"""2""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current154_2,_f,_f,_f])
		psspy.load_chng_4(154,r"""2""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance154_2])

		psspy.load_chng_4(203,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current203,_f,_f,_f])
		psspy.load_chng_4(203,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance203])
		
		psspy.load_chng_4(205,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current205,_f,_f,_f])
		psspy.load_chng_4(205,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance205])
		
		psspy.load_chng_4(3005,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current3005,_f,_f,_f])
		psspy.load_chng_4(3005,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance3005])
		
		psspy.load_chng_4(3007,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current3007,_f,_f,_f])
		psspy.load_chng_4(3007,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance3007])
		
		psspy.load_chng_4(3008,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current3008,_f,_f,_f])
		psspy.load_chng_4(3008,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance3008])
		
		
		psspy.load_chng_4(3019,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f, Change_loading_current3019,_f,_f,_f])
		psspy.load_chng_4(3019,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f, Change_loading_impedance3019])
		
	
		psspy.run(0, Time,1,1,0)
		
		if x == 4800:
			psspy.dist_branch_trip(3011,3019,r"""1""") #Open branch between main grid and microgrid
			psspy.change_channel_out_file(Out_File) #Resume our output file
			
		if x == Converted_Time:
			psspy.dist_branch_close(3011,3019,r"""1""") #Close previously opened branch
			
def PSSE_Arrays2_List(In_Array):
	Temp_1 = ''.join(str(e) for e in In_Array)
	Temp_2 = Temp_1.replace("[", "")
	Temp_2 = Temp_2.replace("]", "")
	In_String = Temp_2.split(', ')
	Out_List = list()
	for x in range(0, len(In_String)):	
		Out_List.append(In_String[x])
	return Out_List

def Load_Duplicate_Fix(In_List):
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
	
def Lists_2_Dicts(Array1, Array2):
	Dictionary = dict(zip(Array1, Array2))
	return Dictionary
	
def loadInfo():	
	ierr, cplxPower = psspy.aloadcplx(-1, 4, 'MVANOM') #Obtain Complex Power of Loads
	ierr, cplxCurrent = psspy.aloadcplx(-1, 4, 'ILNOM') #Obtain Complex Currents of Loads
	ierr, cplImpedance = psspy.aloadcplx(-1, 4, 'YLNOM') #Obtain Complex Impedances of Loads
	ierr, Load_Numbers = psspy.aloadint(-1, 4, 'NUMBER') #Obtain Load Numbers
	ierr, Load_Count = psspy.aloadcount(-1, 4) #Obtain Count of Loads

	return Load_Count, Load_Numbers, cplxPower, cplxCurrent, cplImpedance

def Update_Dict_Vals(Dict2_Update):
	return

	
def ZIP_Loads():
	Load_Count, Load_Numbers, cplxPower, cplxCurrent, cplImpedance = loadInfo() 
	
	Load_Numbers2 = PSSE_Arrays2_List(Load_Numbers)
	
	loadNumbsAndID, Load_ID_List = Load_Duplicate_Fix(Load_Numbers2)

	cplxPower = PSSE_Arrays2_List(cplxPower)
	cplxCurrent = PSSE_Arrays2_List(cplxCurrent)
	cplImpedance = PSSE_Arrays2_List(cplImpedance)

	return cplxPower, cplxCurrent, cplImpedance, Load_Numbers, Load_ID_List


def ZIP_Loadings2_Dicts(cplxPower, cplxCurrent, cplImpedance):
	cplxPower_Dict = Lists_2_Dicts(Load_Numbers, cplxPower)
	cplxCurrent_Dict = Lists_2_Dicts(Load_Numbers, cplxCurrent)
	cplImpedance_Dict = Lists_2_Dicts(Load_Numbers, cplImpedance)
	
	return cplxPower_Dict, cplxCurrent_Dict, cplImpedance_Dict


	
def branchData(zone):
	toBus = []
	fromBus = []
	busInZone = []
	
	busZones = defaultdict(list)
	ierr, arraryFrom = psspy.aflowint(-1, -1, -1, 2, 'FROMNUMBER')
	ierr, arrayTo = psspy.aflowint(-1, -1, -1, 2, 'TONUMBER')
	ierr, busNumber = psspy.abusint(-1, 2, 'NUMBER')
	ierr, busZone = psspy.abusint(-1, 2, 'ZONE')

	
	for y in range(0, len(busNumber[0])):
		busZones[busNumber[0][y]].append(busZone[0][y])
	
	for a in range(0, len(busZones)):
		if busZones[busNumber[0][a]][0] == zone:
			busInZone.append(busNumber[0][a])
	
	for z in range(0, len(arraryFrom[0])):
		if busZones[arraryFrom[0][z]] == busZones[arrayTo[0][z]]:
			pass
		else:
			#print 'Branch between different zones'
			if busZones[arrayTo[0][z]][0] == zone:
				toBus.append(arrayTo[0][z])
				fromBus.append(arraryFrom[0][z])
				#for i in range(0, len(busInZone)):
				
				if arrayTo[0][z] not in busInZone:
					busInZone.append(arrayTo[0][z])
				if arraryFrom[0][z] not in busInZone:
					busInZone.append(arraryFrom[0][z])		
				
			
			if busZones[arraryFrom[0][z]][0] == zone:
				toBus.append(arrayTo[0][z])
				fromBus.append(arraryFrom[0][z])	

				
				if arrayTo[0][z] not in busInZone:
					busInZone.append(arrayTo[0][z])
				if arraryFrom[0][z] not in busInZone:
					busInZone.append(arraryFrom[0][z])			
				

	
	
	return toBus, fromBus, busInZone, busZones, arraryFrom, arrayTo

def outPutChannelsMicro(buses):
	psspy.dyre_new([1,1,1,1],dynamicFile)
	psspy.bsys(1,0,[0.0,0.0],0,[],len(buses),buses,0,[],0,[])
	psspy.chsb(1,0,[-1,-1,-1,1,14,0]) #Bus voltage and angle
	psspy.chsb(1,0,[-1,-1,-1,1,12,0]) #Frequency
	#psspy.chsb(1,1,[-1,-1,-1,1,12,0]) #Frequency
	#psspy.chsb(1,0,[-1,-1,-1,1,13,0]) #Bus voltage
	'''
	psspy.branch_p_and_q_channel([-1,-1,-1,1761,2234],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,2234,2361],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,2234,2361],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,127,157],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,118,142],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,118,1680],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,126,186],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,142,1904],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,1921,1995],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,126,186],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,126,127],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,1667,2038],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,126,186],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,1914,1975],r"""1""","")
	psspy.branch_p_and_q_channel([-1,-1,-1,120,130],r"""1""","")
	'''
#Change (-a+bj) to (-a, b).. any mathematical combination to a correct list representation
def stringToList(complexZIP):
	
	if '-' not in complexZIP:
		if '+' not in complexZIP:
			if 'j' in complexZIP:
				Present_Power = []
				Present_Power.append('0')
				Present_Power.append(complexZIP.strip('j'))

			else:
				Present_Power = []
				Present_Power.append(complexZIP.strip('j'))
				Present_Power.append('0')

		else:			
			Present_Power = complexZIP.strip('()j').split('+')
				
	elif '-' in complexZIP:
		if '+' in complexZIP:
			Present_Power = complexZIP.strip('()j').split('+')

		elif '-' in complexZIP[1]:
			Present_Power = complexZIP.strip('()j').split('-')
			del Present_Power[0]
			Present_Power[0] = '-' + Present_Power[0]
					
		else:
			Present_Power = complexZIP.strip('()j').split('-')
			Present_Power[1] = '-' + Present_Power[1]
			

	return Present_Power
	
def Change_OpPoint(busInZone, Load_Numbs, Load_IDs, cplxPower, cplxCurrent, cplImpedance):
	realPower = []
	reactivePower = []
	
	for x in xrange(0, len(Load_Numbs)):
		
		if Load_Numbs[x] in busInZone:
			Present_Power = stringToList(cplxPower[x])
		
			realPower.append(float(Present_Power[0]))
			reactivePower.append(float(Present_Power[1]))
	
	random.shuffle(realPower)
	random.shuffle(reactivePower)

	
	for y in range(0, len(realPower)):
		psspy.load_chng_4(Load_Numbs[y],Load_IDs[y],[_i,_i,_i,_i,_i,_i],[ realPower[y], reactivePower[y],_f,_f,_f,_f])

def allBuses():
	ierr, iarray = psspy.abusint(-1, 2, 'NUMBER')
	ierr, busZone = psspy.abusint(-1, 2, 'ZONE')
	return iarray, busZone

def zoneBuses(everyBus, busZone, zone):
	busesInZone = []
	busesOutZone = []
	for x in range(0, len(everyBus[0])):
		if busZone[0][x] == zone:
			busesInZone.append(everyBus[0][x])
		else:
			busesOutZone.append(everyBus[0][x])
	
	return busesInZone, busesOutZone
		
def zoneTotals(zone):

	allBus, busZone = allBuses()
	inZone, outZone = zoneBuses(allBus, busZone, zone)
	cplxPower, cplxCurrent, cplImpedance, Load_Numbers, Load_ID_List = ZIP_Loads()
	
	MWinZone, MVARinZone, MWoutZone, MVARoutZone = loadTotals(cplxPower, Load_Numbers, inZone, outZone)
	ierr, testIt = psspy.azonereal(-1, 2, 'PLOAD')
	print MWinZone
	print testIt

def powerTotals():
	pass
	
def loadTotals(cplxPower, Load_Numbers, inZone, outZone):
	totalRealPowerInZone = 0
	totalReactivePowerInZone = 0
	totalRealPowerOutZone = 0 
	totalReactivePowerOutZone = 0

	for x in range(0, len(cplxPower)):
		powerString = stringToList(cplxPower[x]) 
		if Load_Numbers[0][x] in inZone:
			totalRealPowerInZone += float(powerString[0])
			totalReactivePowerInZone += float(powerString[1])
		elif Load_Numbers[0][x] in outZone:
			totalRealPowerOutZone += float(powerString[0])
			totalReactivePowerOutZone += float(powerString[1])	
		else:
			print '================== error, bus neither in nor out of microgrid ===================='
	return totalRealPowerInZone, totalReactivePowerInZone, totalRealPowerOutZone, totalReactivePowerOutZone


def zoneTotalsPsseFunc():
	ierr, zonesMWload = psspy.azonereal(-1, 2, 'PLOAD')
	ierr, zonesMVARload = psspy.azonereal(-1, 2, 'QLOAD')
	ierr, zonesMWgen = psspy.azonereal(-1, 2, 'PGEN')
	ierr, zonesMVARgen = psspy.azonereal(-1, 2, 'QGEN')	
	ierr, zonesMWloss = psspy.azonereal(-1, 2, 'PLOSS')
	ierr, zonesMVARloss = psspy.azonereal(-1, 2, 'QLOSS')		

	return zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss

def inOutMicroGridTotals(zone, zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss):
	microGridRealPower = zonesMWgen[0][zone-1]
	microGridReactivePower = zonesMVARgen[0][zone-1]
	microGridRealLoad = zonesMWload[0][zone-1]
	microGridReactiveLoad = zonesMVARload[0][zone-1]
	microGridRealLoss = zonesMWloss[0][zone-1]
	microGridReactiveLoss = zonesMVARloss[0][zone-1]
	mainGridRealPower = sum(zonesMWgen[0]) - microGridRealPower
	mainGridReactivePower = sum(zonesMVARgen[0]) - microGridReactivePower
	mainGridRealLoad = sum(zonesMWload[0]) - microGridRealLoad
	mainGridReactiveLoad = sum(zonesMVARload[0]) - microGridReactiveLoad
	mainGridRealLoss = sum(zonesMWloss[0]) - microGridRealLoss
	mainGridReactiveLoss = sum(zonesMVARloss[0]) - microGridReactiveLoss

	print '---------------------------Micro grid information---------------------------'
	print 'Total real power: %d MW' %microGridRealPower
	print 'Total reactive power: %d MVAR' %microGridReactivePower
	print 'Total real load: %d MW' %microGridRealLoad
	print 'Total reactive load: %d MVAR' %microGridReactiveLoad
	print 'Total real loss: %d MW' %microGridRealLoss
	print 'Total reactive loss: %d MVAR' %microGridReactiveLoss
	print '---------------------------Main grid information----------------------------'
	print 'Total real power: %d MW' %mainGridRealPower
	print 'Total reactive power: %d MVAR' %mainGridReactivePower
	print 'Total real load: %d MW' %mainGridRealLoad
	print 'Total reactive load: %d MVAR' %mainGridReactiveLoad
	print 'Total real loss: %d MW' %mainGridRealLoss
	print 'Total reactive loss: %d MVAR' %mainGridReactiveLoss	
	
	
	return microGridRealPower, microGridReactivePower, microGridRealLoad, microGridReactiveLoad, microGridRealLoss, microGridReactiveLoss, mainGridRealPower, mainGridReactivePower, mainGridRealLoad, mainGridReactiveLoad, mainGridRealLoss, mainGridReactiveLoss
	
	
def createIsland(toBus, fromBus):
	for x in range(0, len(toBus)):
		#print toBus[x], fromBus	[x]
		psspy.branch_chng(fromBus[x], toBus[x],r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.save(r"""C:\Users\psse\aires\Carter_Case\Poland_Case\Islanded.sav""")

	
def zoneSixDispersion(buses, busZones, zone, toBus, fromBus, fromTotalBus, toTotalBus):
	ierr, type = psspy.abusint(-1, 2, 'TYPE')
	busZoneDictionary = Lists_2_Dicts(buses[0], busZones[0])
	subSet = []
	for x in range(0, len(buses[0])):
		if busZones[0][x] == zone:
			subSet.append(buses[0][x])
	
	for y in range(1, 2): #len(subSet)
		count = 0
		checkedBus = []
		zoneTotals = [0, 0, 0, 0, 0, 0]
		print 'ROOT BUS %d OFFSPRING ARE:' %subSet[y]
		rootZoneTotals = busAdjacentZones(zoneTotals, count, subSet[y], checkedBus, busZoneDictionary, fromTotalBus, toTotalBus)
		print 'For zone 6 bus number %d, zone totals are:' %subSet[y]
		print zoneTotals
		
def busAdjacentZones(zoneTotals, count, bus, checkedBus, busZoneDictionary, fromTotalBus, toTotalBus):
	count += 1
	busAttachements = []
	busAttachementsZones = []
	for x in range(0, len(fromTotalBus[0])):
		if fromTotalBus[0][x] == bus:
			if toTotalBus[0][x] not in busAttachements and toTotalBus[0][x] not in checkedBus:
				busAttachements.append(toTotalBus[0][x])
			
		if toTotalBus[0][x] == bus:
			if fromTotalBus[0][x] not in busAttachements and fromTotalBus[0][x] not in checkedBus:
				busAttachements.append(fromTotalBus[0][x])
	
	checkedBus.append(bus)
	for a in range(0, len(busAttachements)):
		checkedBus.append(busAttachements[a])
		
		
	
	for z in range(0, len(busAttachements)):
		busAttachementsZones.append(busZoneDictionary[busAttachements[z]])

	print busAttachements
	print busAttachementsZones
	
	for b in range(0, len(zoneTotals)):
		zoneTotals[b] += busAttachementsZones.count(b + 1)
	
	
	
	
	for y in range(0, len(busAttachements)):
		#print busZoneDictionary[busAttachements[y]]
		
		if count < 5:
			print 'OFFSPRING OF INTERMEDIATE BUS %d' %busAttachements[y]
			zoneTotals = busAdjacentZones(zoneTotals, count, busAttachements[y], checkedBus, busZoneDictionary, fromTotalBus, toTotalBus)
		
	return zoneTotals

	
def runFlatStart(dataFileName, buses, outFile):
	okToCont = True
	
	psspy.lines_per_page_one_device(1,10000)   
	psspy.progress_output(2,dataFileName,[0,0])
	rarray, okSolution = Solve_Steady()
	Convert_Dynamic()
	outPutChannelsMicro(buses)
	flatStart(outFile, 2)
	psspy.delete_all_plot_channels()
	
	
	operatingToBuses, operatingFromBuses, otherRelaysOperating = parseDataFile(dataFileName + '.DAT')
	

	
	existsNonConverge = checkFileForPhrase('Network not converged', dataFileName + '.DAT')
	if existsNonConverge:
		print 'network does not converge within 2 seconds of flat start'
		okToCont = False
	if okSolution == 0:
		print 'voltage violation when interconnected'
		okToCont = False
	if otherRelaysOperating:
		okToCont = False
		print 'other relays besides overcurrent are operating'
	
	
	return okToCont, operatingToBuses, operatingFromBuses
	
def newInitialConditionsOkCheck(inFile, saveTo, slackBuses, zone, fixOpPoint):
	''' Take in our network, modify the initial condions and check to see if the case will be okay to run islanding on.  We 
	check that the case is stable for a certain period of time when interconnected as well as islanded (No relay operations and 
	converges).
	'''
	
	initializeCase(inFile) #initialize our saved case 
	toBus, fromBus, busInZone, busZones, fromTotalBus, toTotalBus = branchData(zone) #Get information on branches (toBus, fromBus) are for the branches connecting to the zone.  
	cplxPower, cplxCurrent, cplImpedance, Load_Numbers, Load_ID_List = ZIP_Loads()
	changeInitConds(Load_Numbers[0], Load_Numbers[0], Load_ID_List, cplxPower, cplxCurrent, cplImpedance)	#Makes the changes to initial conditions
	psspy.save(os.path.join(saveTo, 'tempFileStart.sav')) #Make a copy of the new save case
	
	
	outputTempInt = os.path.join(saveTo, 'outputTempInt')
	outputTempIsl = os.path.join(saveTo, 'outputTempIsl')
	outputTempIntDat = os.path.join(saveTo, 'outputTempInt.DAT')
	outputTempIslDat = os.path.join(saveTo, 'outputTempIsl.DAT')
	tempOutputInt = os.path.join(saveTo, 'tempFlatStartInt.out')
	tempOutputIsl = os.path.join(saveTo, 'tempFlatStartIsl.out')
	
	interconnectedOk, operatingToBuses, operatingFromBuses = runFlatStart(outputTempInt, busInZone, tempOutputInt) #Check if the interconnection case is okay.
	psspy.progress_output(1,'',[0,0]) #Terminate printing output to a file
	if len(operatingFromBuses) > 0 and len(operatingToBuses) > 0:
		interconnectedOk = False
		print 'Relays operating during flat start'
		
	#If the interconnection case was okay, move on to islanded case
	if interconnectedOk or fixOpPoint:
		initializeCase(os.path.join(saveTo, 'tempFileStart.sav'))	#Reinitialze our case since the old one has been converted to dynamics
		islandCaseNoDynamics(zone, slackBuses)
		islandedOk, operatingToBuses, operatingFromBuses = runFlatStart(outputTempIsl, busInZone, tempOutputIsl) #Run the islanded case
		operatingToBuses, operatingFromBuses = removeBranchFromPairList(operatingToBuses, operatingFromBuses, toBus, fromBus)
		if len(operatingFromBuses) > 0 and len(operatingToBuses) > 0:
			interconnectedOk = False
			print 'Relays operating during flat start'
		
		psspy.progress_output(1,'',[0,0]) #Terminate printing output to a file
		os.remove(tempOutputInt)
		os.remove(tempOutputIsl)
		if not fixOpPoint:
			os.remove(outputTempIntDat)
			os.remove(outputTempIslDat)
			
		#If the islanded case is okay return True
		if islandedOk:
			return True, toBus, fromBus, busInZone
	#If the interconnection case is not okay we will return False
	else:
		os.remove(tempOutputInt)
		os.remove(outputTempIntDat)
		return False, toBus, fromBus, busInZone
		
	
	#return false since our only condition of passing is if both islanded and interconnection cases are stable
	return False, toBus, fromBus, busInZone
		

	
def islandZoneDynamics(zone, slackBuses, fileTitle, countCases, inFile, saveTo):
	'''This will run our islanding simulation for a certain network.  It will only occur if the voltages are within tolerable ranges to start.
	
	
	'''
	if not os.path.exists(saveTo):
		os.makedirs(saveTo)

	outputDataFile = os.path.join(saveTo, 'outputData.DAT')
	outputFile = os.path.join(saveTo, fileTitle + '.out')
	saveFileCompleted = os.path.join(saveTo, fileTitle + '.sav')
	savFileInit = os.path.join(saveTo, fileTitle + '.sav')
	saveFileBeforeRecon = os.path.join(saveTo, fileTitle + 'BeforeRecon' + '.sav')
	
	okayToProceed, toBus, fromBus, busInZone = newInitialConditionsOkCheck(inFile, saveTo, slackBuses, zone, False)
		
	if okayToProceed:
		initializeCase(os.path.join(saveTo, 'tempFileStart.sav'))	
		psspy.lines_per_page_one_device(1,10000)   
		psspy.progress_output(2,outputDataFile,[0,0])
		Convert_Dynamic()
		outPutChannelsMicro(busInZone)
		removeFileFlag = Islanding(busInZone, 2, 3, 25, 85,outputFile, toBus, fromBus, saveFileBeforeRecon, outputDataFile, saveTo)
		psspy.delete_all_plot_channels()
		if removeFileFlag == 0:
			os.remove(outputFile)
		else:
			psspy.save(saveFileCompleted)
			countCases += 1
		
	os.remove(os.path.join(saveTo, 'tempFileStart.sav'))
	

	return countCases
def islandCaseNoDynamics(zone, slackBuses):
	if zone == 1:
		psspy.bus_chng_3(185,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
	else:
		psspy.bus_chng_3(slackBuses[zone - 1],[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
	toBus, fromBus, busInZone, busZones, fromTotalBus, toTotalBus = branchData(zone)
	createIsland(toBus, fromBus)
	
	
	ierr, buses = psspy.tree(1, 1)
	while buses != 0:
		ierr, buses = psspy.tree(2, 1)	
	
def growTreeOfBusesInZone(zone):
	initializeCase()
	toBus, fromBus, busInZone, busZones, fromTotalBus, toTotalBus = branchData(zone)
	buses, busZones = allBuses()
	zoneSixDispersion(buses, busZones, zone, toBus, fromBus, fromTotalBus, toTotalBus)

def addLineRelays(lineIDs, fromBranch, toBranch, inputFile, relayType, Load_Numbers, branchRateArevised, branchRateBrevised, branchRateCrevised):
	#print fromBranch
	#print toBranch
	f = open(inputFile, 'r')
	contents = f.readlines()
	f.close()
	f = open(inputFile, 'w')
	print len(contents)
	for x in range(0, len(contents)):
		f.write(contents[x])
		if '1' in contents[x] and '16' in contents[x] and 'TIOCR1' in contents[x]:
		
			for y in range(0, len(fromBranch[0])):
				
				branchArate = lineValues(lineIDs[y], fromBranch[0][y], toBranch[0][y], fromBranch, toBranch, branchRateArevised)
				branchBrate = lineValues(lineIDs[y], fromBranch[0][y], toBranch[0][y], fromBranch, toBranch, branchRateBrevised)
				branchCrate = lineValues(lineIDs[y], fromBranch[0][y], toBranch[0][y], fromBranch, toBranch, branchRateCrevised)
				one = contents[x].replace(' ', '').strip('\n').split(',')
				#print one
				one[0] = str(fromBranch[0][y])
				one[2] = str(toBranch[0][y])
				#print fromBranch[0][y], toBranch[0][y], lineMVA
				if toBranch[0][y] in Load_Numbers[0]:
					one[6] = str(toBranch[0][y])
				elif fromBranch[0][y] in Load_Numbers[0]:
					one[6] = str(fromBranch[0][y])
					
				else:
				
					one[8] = str(fromBranch[0][y])
					one[9] = str(toBranch[0][y])
					one[6] = ' '
				
				if lineIDs[y] == 1:
					one[3] = "'1'"
				elif lineIDs[y] == 2:
					one[3] = "'2'"
					
				one[17] = str(branchArate/100)
				one[19] = str(branchBrate/100)
				one[21] = str(branchCrate/100)
				one[22] = str(0.15)
				one[24] = str(0.15)
				one[26] = str(0.15)
				one[28] = str(0.15)
				one[30] = str(0.15)
				one[23] = str(branchCrate/100)
				one[25] = str(branchCrate/100)
				one[27] = str(branchCrate/100)
				one[29] = str(branchCrate/100)
				#one[23] = str(10000)
				#one[25] = str(10000)
				#one[27] = str(10000)
				#one[29] = str(10000)
				
					
				one[-1] = '0.1/'
				oneString = ', '.join(one)
				f.write('\n')
				f.write(oneString)
		

def lineValues(IDnumber, currentFrom, currentTo, fromBranch, toBranch, linePU):
	
	
	for x in range(0, len(fromBranch[0])):
		if fromBranch[0][x] == currentFrom and toBranch[0][x] == currentTo and IDnumber == 1:
			val = linePU[0][x]
			break
		elif fromBranch[0][x] == currentFrom and toBranch[0][x] == currentTo and IDnumber ==2:
			val = linePU[0][x]
			break
	
	return val

def deleteDuplicateLineInstances(fromBranch, toBranch):
	newFromBranches = []
	newToBranches = []
	badIndex = []
	goodIndex = []
	for x in range(0, len(fromBranch[0])):


		flag = 1
		for y in range(0, len(fromBranch[0])):
			if fromBranch[0][x] == toBranch[0][y] and toBranch[0][x] == fromBranch[0][y]:
				badIndex.append(y)
				
		if x not in badIndex:
			goodIndex.append(x)
			newFromBranches.append(fromBranch[0][x])
			newToBranches.append(toBranch[0][x])
	
	#print goodIndex
	return [newFromBranches], [newToBranches], goodIndex

def Return_Load_Info():	
	ierr, Complex_Power = psspy.aloadcplx(-1, 4, 'MVANOM') #Obtain Complex Power of Loads
	ierr, Complex_Current = psspy.aloadcplx(-1, 4, 'ILNOM') #Obtain Complex Currents of Loads
	ierr, Complex_Impedance = psspy.aloadcplx(-1, 4, 'YLNOM') #Obtain Complex Impedances of Loads
	ierr, Load_Numbers = psspy.aloadint(-1, 4, 'NUMBER') #Obtain Load Numbers
	ierr, Load_Count = psspy.aloadcount(-1, 4) #Obtain Count of Loads

	return Load_Count, Load_Numbers, Complex_Power, Complex_Current, Complex_Impedance
	
def deleteBadIndices(goodIndeces, listToModify):

	newList = []
	for x in range(0, len(listToModify)):
		if x in goodIndeces:
			#print x
			newList.append(listToModify[x])
		
	return [newList]

def addLineRelaysTotal():
	inputFile = 'newPolandDynamicsCopy.dyr'
	relayType = 'TIOCR1'
	zone = 5
	initializeCase()
	notUsedOne, notUsedTwo = Solve_Steady()
	Load_Count, Load_Numbers, Complex_Power, Complex_Current, Complex_Impedance = Return_Load_Info() 
	toBus, fromBus, busInZone, busZones, fromTotalBus, toTotalBus = branchData(zone)
	
	newFromBranches, newToBranches, keepIndex = deleteDuplicateLineInstances(fromTotalBus, toTotalBus)
	ierr, branchRateA = psspy.aflowreal(-1, -1, -1, 2, 'RATEA')
	ierr, branchRateB = psspy.aflowreal(-1, -1, -1, 2, 'RATEB')
	ierr, branchRateC = psspy.aflowreal(-1, -1, -1, 2, 'RATEC')
	
	branchRateA = deleteBadIndices(keepIndex, branchRateA[0])
	branchRateB = deleteBadIndices(keepIndex, branchRateB[0])	
	branchRateC = deleteBadIndices(keepIndex, branchRateC[0])

	
	#divideLists(branchMVA, branchRateAprcnt)
	lineIDs = lineIDListCreation(newFromBranches[0], newToBranches[0])
	
	addLineRelays(lineIDs, newFromBranches, newToBranches, inputFile, relayType, Load_Numbers, branchRateA, branchRateB, branchRateC)

def lineIDListCreation(newFromBranches, newToBranches):
	listOfLineIDs = []
	listOfLineIDs.append(1)
	for x in range(1, len(newFromBranches)):
		
		
		if newFromBranches[x] == newFromBranches[x-1] and newToBranches[x] == newToBranches[x-1]:

			listOfLineIDs.append(2)
		
		else:
			listOfLineIDs.append(1)
				

	
	return listOfLineIDs


def getBranchFlows(caseName,zone,desiredToBuses,desiredFromBuses):

	initializeCase(caseName)
	notUsedOne, notUsedTwo = Solve_Steady()
	Load_Count, Load_Numbers, Complex_Power, Complex_Current, Complex_Impedance = Return_Load_Info() 
	toBus, fromBus, busInZone, busZones, fromTotalBus, toTotalBus = branchData(zone)
	
	newFromBranches, newToBranches, keepIndex = deleteDuplicateLineInstances(fromTotalBus, toTotalBus)
	ierr, branchPctRateA = psspy.aflowreal(-1, -1, -1, 2, 'PCTRATEA') #RATEA
	ierr, branchRateA = psspy.aflowreal(-1, -1, -1, 2, 'RATEA') #RATEA	
	
	
	ierr, branchRateB = psspy.aflowreal(-1, -1, -1, 2, 'PCTRATEB')
	ierr, branchRateC = psspy.aflowreal(-1, -1, -1, 2, 'PCTRATEB')
	
	
	branchPctRateA = deleteBadIndices(keepIndex, branchPctRateA[0])
	branchRateA = deleteBadIndices(keepIndex, branchRateA[0])
	#branchRateB = deleteBadIndices(keepIndex, branchRateB[0])	
	#branchRateC = deleteBadIndices(keepIndex, branchRateC[0])
	
	branchNewRate = []
	
	for x in range(0, len(newFromBranches[0])):
		#print newFromBranches[0][x], newToBranches[0][x], branchRateA[0][x], branchRateB[0][x], branchRateC[0][x]
	
		
		for y in range(0, len(desiredToBuses)):

			if int(desiredToBuses[y]) == int(newToBranches[0][x]) and int(desiredFromBuses[y]) == int(newFromBranches[0][x]):
				print desiredToBuses[y], desiredFromBuses[y], y, x
				branchNewRate.append((branchRateA[0][x]/100)*(branchPctRateA[0][x]/100)*1.5)
				#print desiredToBuses[y], desiredFromBuses[y], branchRateA[0][x], branchRateB[0][x], branchRateC[0][x]
		
	return desiredToBuses, desiredFromBuses,branchNewRate
	
def checkIfPairExists(checkTo, checkFrom, toBus, fromBus):
	'''This will check to see if the toBus and associated from Bus already exists in a corresponding
	checkTo and checkFrom lists
	'''
	#For each item in checkTo
	for x in range(0, len(checkTo)):
		#If the list checkTo already has toBus in it AND checkFrom already has fromBus at the same index then return True since the pair already exists
		if checkTo[x] == toBus and checkFrom[x] == fromBus:
			return True
			
	#Pair doesn't exist
	return False

def removeBranchFromPairList(toBuses, fromBuses, toBusException, fromBusException):
	'''Function exclusively for removing certain branches from a toBus list and fromBus list pair
	'''

	newToBuses = []
	newFromBuses = []
	for x in range(0, len(toBuses)):
		flag = True
		for y in range(0, len(toBusException)):
			if str(toBuses[x]) == str(toBusException[y]) and str(fromBuses[x]) == str(fromBusException[y]):
				flag = False
		
		if flag:
			newToBuses.append(toBuses[x])
			newFromBuses.append(fromBuses[x])
	

	return newToBuses, newFromBuses
	
	
	
def parseDataFile(dataFile):
	'''This checks the .DAT file from PSSE to check for operating TIOCR1 relays during flat start conditions.
	We return the branches in the form of the connecting to and from buses.
	'''
	

	badFromBuses = [] #Branch from buses
	badToBuses = [] #Branch to buses 
	otherRelaysOperating = False #flag to check if other types of relays are operating
	
	#Open the data file 
	with open(dataFile) as f:
		content = f.readlines() #Read the content of the data file
	
	#For each line in the file
	for line in content:
		#If the line contains the keyword 'PICKUP'
		if 'PICKUP' in line:
			#If the line contains the keyword 'TIOCR1' representing the overcurrent relay
			if '-0.017' in line:
				#Make sure that the relay is operating before 0 seconds
				if 'TIOCR1' in line:
					badRelayOperation = " ".join(line.split()) #We create a list then join it again to strip the excess white spaces
					badRelayOperation = badRelayOperation.split(' ') #Resplit based on white spaces 
					
					
					#These next two lines will take the line list that contains the operating relays and find the indices#
					#of 'TO' and 'FROM' and get the words immediately after each as they will be the associated buses.#					
					toBus = badRelayOperation[badRelayOperation.index('TO') + 1] 
					fromBus = badRelayOperation[badRelayOperation.index('FROM') + 1]
				
					#If the toBus fromBus pair does not exist, add toBus fromBus pair
					if not checkIfPairExists(badToBuses, badFromBuses, toBus, fromBus):
						badFromBuses.append(fromBus) 
						badToBuses.append(toBus)
		
				else:
					otherRelaysOperating = True
		
	
	
	
	
	return badToBuses, badFromBuses, otherRelaysOperating

	
	

def parseDynamicsFile(relayType, toBus, fromBus, dyrFile, branchNewRate):
	f = open(dyrFile, 'r')
	contents = f.readlines()
	f.close()
	f = open(dyrFile, 'w')
	print len(contents)
	for x in range(0, len(contents)):
		if relayType in contents[x]:
			desiredList = contents[x].replace(' ', '').strip('\n').split(',')
			flag = True
			for y in range(0, len(toBus)):
				if str(fromBus[y]) == desiredList[0] and str(toBus[y]) == desiredList[2]:
					desiredList[17] = str(branchNewRate[y])
					desiredList[19] = str(1.0)
					desiredList[21] = str(1.25)
					desiredList[23] = str(1.375)
					desiredList[25] = str(1.5)
					desiredList[27] = str(1.625)
					
					desiredString = ', '.join(desiredList)
					f.write(desiredString)
					f.write('\n')
					flag = False
			if flag:
				f.write(contents[x])
		else:
			f.write(contents[x])
			
			
def runIslandingSimulationWrapper(zone, slackBuses, saveTo, infile, simsToRun):
	countCases = 0
	while countCases != simsToRun:
		countCases = islandZoneDynamics(zone, slackBuses, 'polandIslandingTestControl' + '_' + str(countCases), countCases, infile, saveTo)
	
	
def main():

	pass
	

if __name__ == "__main__": 
	main()	



