from __future__ import division
import os,sys
import os.path


sys.path.append(r"C:\Program Files (x86)\PTI\PSSEXplore33\PSSBIN") #Give the path to PSSBIN to imoport psspy
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSEXplore33\PSSBIN;" #Tell PSSE where "itself" is
                      + os.environ['PATH'])
					  
import psspy
import redirect
import dyntools
import pssplot
import csv
import pandas as pd
#import matplotlib
import numpy as np					  
import copy as cp
import math
import gc
import heapq



def outfile2python(outfile):
	data = dyntools.CHNF(outfile)
	x, y, z = data.get_data()
	del data
	return x,y,z

def Python2Csv(x,y,z,outfile,Toggle,Reconnect_Time):
	s1 = pd.Series(y)
	s2 = pd.Series(z)
	s1 = list(s1)
	stability = 1
	#print 'Done with series'
	ETRM_Count = 0
	CSV_Name = (outfile.split('.'))[0]
	df1 = pd.DataFrame(z, columns=None)
	#print 'df1 done'
	
	for x in xrange(0,len(s1) - 1):
		s1[x] =  (s1[x].split(' ['))[0]
		s1[x] =  (s1[x].split('['))[0]
		if s1[x].count('ETRM'):
			ETRM_Count += 1
	
	#print 'Counted ETRM'
	df1.columns = s1
	

				
	CheckTime = 25
	data_lengths = len(df1.index)
	start_point = len(df1.index) - CheckTime*120
	start_point2 = len(df1.index) - 2*CheckTime*120
	start_point3 = ((len(df1.index) - Reconnect_Time*120)/2) + Reconnect_Time*120
	#print start_point
	
	df2 = df1.ix[start_point:]
	df3 = df1.ix[start_point2:]
	df4 = df1.ix[start_point3:]
	percent_errorTOL = 10
	for z in range(0, len(s1) - 1):
		if 'VOLT' in s1[z]:
			#print (df2[s1[z]].mean() - df2[s1[z]].min(axis=0))/df2[s1[z]].mean()
			if (df2[s1[z]].max(axis=0) - (df2[s1[z]].max(axis=0) + df2[s1[z]].min(axis=0))/2)/(df2[s1[z]].max(axis=0) + df2[s1[z]].min(axis=0))/2 > percent_errorTOL/100 or ((df2[s1[z]].max(axis=0) + df2[s1[z]].min(axis=0))/2 - df2[s1[z]].min(axis=0))/(df2[s1[z]].max(axis=0) + df2[s1[z]].min(axis=0))/2 > percent_errorTOL/100:

				stability = 0

			if df2[s1[z]].max(axis=0) >= df3[s1[z]].max(axis=0):
				stability = 0
			
			if df4[s1[z]].max(axis=0) > 1.4 or df4[s1[z]].min(axis=0) < 0.6:
				stability = 0
				
	print stability
	#Unwrap_Data(df1, s1)
	#print 'finished unwrapping'
	df1.to_csv(CSV_Name + '.csv', index = False)
	print 'Done writing %s' %CSV_Name + '.csv'
	del df1
	del s1
	del s2
	del df2
	del df3
	del df4
	gc.collect()
	return ETRM_Count, CSV_Name + '.csv', stability
	
	
	
	
def Unwrap_Data(Data_Frame, DFlabels):

	for y in range(0,len(DFlabels)):
		if DFlabels[y].count('ANGL'):
			for x in xrange(0,len(Data_Frame.ix[:, DFlabels[0]])):
		
				Data_Frame.ix[:, DFlabels[y]][x] = math.asin(math.sin((math.pi/180)*Data_Frame.ix[:, DFlabels[y]][x]))*(180/math.pi)

			
			#for x in xrange(0,len(Data_Frame.ix[:, DFlabels[0]])):
				
				
				#if Data_Frame.ix[:, DFlabels[y]][x] > 360 or Data_Frame.ix[:, DFlabels[y]][x] < -360:
					
	
				
	#print len(Data_Frame.ix[:, DFlabels[0]])
		
	#print Data_Frame.ix[:, DFlabels[0]]
	
	
	
	
	
def Sample_CSV(Frequency, df):
	s = 0
	df = df[df['Time(s)'] >= 0]
	df = df.reset_index()
	total_rows = len(df.axes[0])
	Time = df['Time(s)'].iget(-1)
	Samples = Time/Frequency
	PMU_Data_Step = total_rows / Samples
	
	s = PMU_Data_Step
	x = df.loc[[0]]
	y = x
	print 'Working on sampling total data'
	for i in range (1,total_rows):
		t = int(s)
		if t > total_rows:
			break
		else:
			y = y.append(df.loc[t],ignore_index=True)
			s = s + PMU_Data_Step
						
	del y['index']
	y.to_csv('All_Data_Sampled.csv',index=False)
	print 'Done writing All_Data_Sampled.csv'
	return y

def Break_CSV(list, df, status):
	for x in range(0, len(list) - 1):
		w = df.filter(regex='Time.*')
		w = pd.concat([w, df.filter(regex=list[x])], axis=1)
		w.to_csv('Data_' + list[x] + status + '.csv',index=False)
		print 'Done writing %s' %'Data_' + list[x] + status + '.csv'
	
#####NOTE: Flag = 1 classifies an unstable reconnection!! Classification appended to start of file #########
def Classify_Data(CSVFile, ETRM_Count, Reconnect_Time):
	#CurrentCSV = np.genfromtxt(CSVFile,delimiter=',')
	CurrentCSV = []
	count = 0
	with open(CSVFile, 'r') as file:
		r = csv.reader(file, delimiter=',')
		for row in r:
			count += 1
			if count > Reconnect_Time*120:
				CurrentCSV.append(row)
				count += 1
				#print count
	

	flag = 0
	count = 0
	for x in xrange(0, ETRM_Count): 
		if any(CurrentCSV[x] < 0.65) or any(CurrentCSV[x] > 1.4):
			flag = 1
	flag_str = str(flag)
	

	target = open(CSVFile, 'r')
	lines = target.readlines()
	target.close()
	target = open(CSVFile, 'w')
	for line in lines:
		flag = flag_str
		count += 1
		if count == 1:
			flag = 'Classification'
			newlist = ",".join((flag, line))
		line = ",".join((flag, line))
		target.write(line)
	
	return CSV_Data, CurrentCSV, newlist, flag
	
	
	
	
def Create_Features(CSV_Data, CSV_DataTP, newlist, flag, Reconnect_Time):
	Time_step = CSV_DataTP[-1][1] - CSV_DataTP[-1][0]
	x = 0
	index_list = []
	Feature_List = []
	while CSV_DataTP[-1][x] < Time_step:
		x += 1
	Feature_Locations = x + 2 + Reconnect_Time/Time_step
	#print CSV_Data[Feature_Locations]
	newlist = newlist.strip('\n').split(',')
	for y in xrange(0, len(newlist)):
		if newlist[y].count('VOLT'):
			index_list.append(y)
		if newlist[y].count('ANGL'):
			index_list.append(y)
	for z in index_list:
		Feature_List.append(CSV_Data[Feature_Locations][z-1])
	Feature_List.append(flag)
	
	return Feature_List

def Create_Total_TrainingData(Current_Example):
	Current_Example = ','.join(map(str, Current_Example))
	#print Current_Example
	fd = open('Training_Data_Total.csv','a')
	fd.write(Current_Example)
	fd.write('\n')
	fd.close()
		
	
	
def NewTrainingData_Creation(CSVFILE, stability, Reconnect_Time):
	counter = 0
	Features = []
	Features_Names = []
	with open(CSVFILE, 'r') as file:
		r = csv.reader(file, delimiter=',')
		for row in r:
			counter += 1
			if counter == 1:
				Column_Headers = row
			if counter == (Reconnect_Time*120) + 9:
				Features_Temp = row
	del r
	for x in range(0, len(Column_Headers)):
		if 'ANGL' in Column_Headers[x] or 'VOLT' in Column_Headers[x] or 'Time(s)' in Column_Headers[x]:
			Features_Temp[x] = float(Features_Temp[x])
			Features.append(Features_Temp[x])
			Features_Names.append(Column_Headers[x])
	
	
	#print len(Features_Names)
	#print len(Features)
	
	for y in range(0, len(Features_Names)):
		if 'ANGL' in Features_Names[y]:
			Features[y] = math.asin(math.sin((math.pi/180)*Features[y]))*(180/math.pi)
	Features.append(stability)
	
	return Features


def unwrapList(listToUnwrap):
	returnedList = []
	for x in range(0, len(listToUnwrap)):
		placeHolder = listToUnwrap[x]%360
		if placeHolder < -180:
			placeHolder += 360
		elif placeHolder > 180:
			placeHolder -= 360
		returnedList.append(placeHolder)
		
	return returnedList
	
def returnTimeStepBeforeReconnection(Reconnect_Time, y, z):
	PMUdataLabels = []
	voltageKeys = []
	angleKeys = []
	currentTimeIndex = 0
	voltageBeforeReconnect = []
	angleBeforeReconnect = []
	
	#print z['time'][-1]
	while z['time'][currentTimeIndex] <= Reconnect_Time:
		currentTimeIndex += 1 
		#print z['time'][currentTimeIndex]
 

	
	for a in range(1, len(y)):
		if 'VOLT' in y[a].split('[')[0]:
			PMUdataLabels.append(y[a].split('[')[0])
			voltageKeys.append(a)
		elif 'ANGL' in y[a].split('[')[0]:
			PMUdataLabels.append(y[a].split('[')[0])
			angleKeys.append(a)
			
	for b in voltageKeys:
		voltageBeforeReconnect.append(z[b][currentTimeIndex])
	for c in angleKeys:
		angleBeforeReconnect.append(z[c][currentTimeIndex])
	
	

	angleBeforeReconnect = unwrapList(angleBeforeReconnect)
	
	Total_Variables = voltageBeforeReconnect + angleBeforeReconnect
	
	return Total_Variables, currentTimeIndex

def appendLabel(Features, MasterCsvfile, currentExample, stabilityFlag):
	f = open(MasterCsvfile, 'r')
	contents = f.readlines()
	f.close()
	
	
	'''
	for x in range(0, len(contents)):
		if currentExample in contents[x]:
			caseLabel = contents[x].strip().split(',')[1]

	if int(caseLabel) == 0:
		Features.append(caseLabel)
		#print 'PSS/e has declared instability for case: %s' %currentExample
		#print ' '
	elif int(caseLabel) != stabilityFlag:
		Features.append(stabilityFlag)
		print 'PSS/e failed to correctly classify, backup stability check reassigned for case: %s' %currentExample
		print ' '
	else:
		Features.append(stabilityFlag)
		#print 'PSS/e and backup stability check agree for case: %s' %currentExample
		#print ' '
	'''
	Features.append(stabilityFlag)
	return Features
	
def analyzeData(y, z, desiredEndTimeAnalysis, timeAfterReconnectCheck, Reconnect_Time, currentTimeIndex):
	timeStepAfter = timeAfterReconnectCheck*120
	keyForFrequency = []
	keysForVoltage = []
	voltageLoc = []
	endStepsData = desiredEndTimeAnalysis*120
	totalSteps = len(z[1])
	maxTime = z['time'][-1]
	timeStep = 0
	#a = [0, 2, 33, 1, 22]
	#print sorted(range(len(a)), key=lambda i: a[i])[-2:]

	
	for a in range(1, len(y)): #len(y)
		if 'FREQ' in y[a].split('[')[0]:
			keyForFrequency.append(a)
	
	for c in range(1, len(y)): #len(y)
		if 'VOLT' in y[c].split('[')[0]:
			keysForVoltage.append(c)
			voltageLoc.append(y[c].split('[')[0])
	
	
	for b in keysForVoltage:
		lastFrame = z[b][currentTimeIndex + int(2*timeStepAfter):currentTimeIndex + int(4*timeStepAfter)]


		lastSubFrame = z[b][-int(endStepsData):]
		
		if (max(lastSubFrame) - (max(lastSubFrame) + min(lastSubFrame))/2) > (max(lastFrame) - (max(lastFrame) + min(lastFrame))/2):
			
			#print voltageLoc[keysForVoltage.index(b)]
			if int(voltageLoc[keysForVoltage.index(b)].split(' ')[1]) < 100:
				print voltageLoc[keysForVoltage.index(b)]
				return 0
				
			
	'''	
	for b in keyForFrequency:
		#lastFrameDividedByTwo = z[b][int(-endStepsData/2):]
		lastFrame = z[b][-endStepsData:]
		if max(lastFrame) > 0.01 or min(lastFrame) < -0.01:
			return 0
	'''
	return 1
def topLastThreeLocalMax(endList, maxTime, z, totalSteps):
	topVals = []
	topValsIndex = []
	top3ValsIndex = []
	flag = 0
	
	for x in range(1, len(endList)):
		if endList[x] >= endList[x-1]:
			flag = 1
		elif flag == 1:
			topVals.append(endList[x])
			topValsIndex.append(x)
			flag = 0
	top3ValsIndex = sorted(range(len(topVals)), key=lambda i: topVals[i])[-4:]
	#print top3ValsIndex[-1]
	#print topVals
	for y in top3ValsIndex:
		print z[147][totalSteps - len(endList) + topValsIndex[y]]
		print z['time'][totalSteps - len(endList) + topValsIndex[y]]
	
	return topValsIndex, topVals
def main():
	Reconnect_Time = 25
	Toggle = 1 #Skip unneeded data, just grab the volts + angles if 1.
	MasterCsvfile = 'MasterConvergenceFile.csv'
	desiredEndTimeAnalysis = 10
	timeAfterReconnectCheck = 5

	for s in xrange(0,20):
		#print 'at: %i' %s
		x_s = str(s)
		outfile = 'polandIslandingTestControl' + '_' + x_s + '.out'
		currentExample = 'polandIslandingTestControl' + x_s + ','
		if os.path.isfile(outfile):
			x, y, z = outfile2python(outfile)
			Features, currentTimeIndex = returnTimeStepBeforeReconnection(Reconnect_Time, y, z)
			
			stabilityFlag = analyzeData(y, z, desiredEndTimeAnalysis, timeAfterReconnectCheck, Reconnect_Time, currentTimeIndex)
			if stabilityFlag == 0:
				print 'polandIslandingTestControl' + x_s + '.out'
	
			appendLabel(Features, MasterCsvfile, currentExample, stabilityFlag)
			
			#Create_Total_TrainingData(Features)
			
			del x
			del y
			del z			
			del Features
	print 'Features appended to total_features csv file'	
	
	'''
			CSV_Data, CurrentCSV, newlist, flag = Classify_Data(CSV_Name, ETRM_Count, Reconnect_Time)
			Training_Feats_Targ = Create_Features(CSV_Data, CurrentCSV, newlist, flag, Reconnect_Time)
			Create_Total_TrainingData(Training_Feats_Targ)
	'''
if __name__ == "__main__":
    main()	




























#Outdated and clunky version of converting .out to .csv
# def outfile2csv():
	# out_file = 'Reg_Angle'
	# xlsoutput= out_file+'.xlsx'
	# out_file    = out_file+'.out'
	# xlsresult= dyntools.CHNF(out_file) 
	# dyntools.CHNF.xlsout(xlsresult, channels=[], show=False, xlsfile=xlsoutput, outfile=out_file)
	# data_xls = pd.read_excel(xlsoutput, 'Sheet1', index_col=None)
	# data_xls.to_csv('All_Data.csv')
	# df = pd.read_csv('All_Data.csv')
	# df = df.reindex()
	# print df
	# df.to_csv('All_Data.csv', index = False)

