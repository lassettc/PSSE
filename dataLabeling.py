import os,sys
import csv
import random
import copy as cp



def writeData(masterDataFile, totalMeasurements):
	currentExample = ','.join(map(str, totalMeasurements))
	fd = open(masterDataFile,'a')
	fd.write(currentExample)
	fd.write('\n')
	fd.close()
	
	
	
def openCsv(csvFile):
	with open(csvFile, 'rb') as f:
		reader = csv.reader(f)
		dataList = list(reader)
	
	return dataList




	
def stabilityWrapper(convergenceFile, pmuDataFile, numberPerClass, csvFileName):
	dataListConvergence = openCsv(convergenceFile)
	dataListPmus = openCsv(pmuDataFile)

	for x in range(0, len(dataListPmus)-1):	
		
		dataLabelPmuFile = dataListPmus[x].pop(0)
		dataLabelConvergenceFile = dataListConvergence[x].pop(0)
		dataLabelPmuFile = dataLabelPmuFile.split('.')[0]
		
		
		if dataLabelPmuFile == dataLabelConvergenceFile:
			dataList = parseConvergenceFile(dataListConvergence[x])
			#print dataList
			stability = getStabilityLabel(dataList)
			#print stability
			dataListPmus[x].append(stability)
			
			
	random.shuffle(dataListPmus)		
	dataCollected = False
	count = 0
	zerosCount = 0
	onesCount = 0
	for y in range(0, len(dataListPmus)):
		
		if dataListPmus[y][-1] == 0:	
			if zerosCount < 50:
				#writeData(csvFileName, dataListPmus[y])
				zerosCount += 1 
		elif dataListPmus[y][-1] == 1:
			if onesCount < 50:
				#writeData(csvFileName, dataListPmus[y])
				onesCount += 1 	
		
		if onesCount == 50 and zerosCount == 50:
			break
		
	'''
				if random.uniform(0, 1) > 0:
					if random.uniform(0, 1) > 0.75:
						writeData('test.csv', dataListPmus[x])
					else:
						writeData('train.csv', dataListPmus[x])
				
			else:
				if random.uniform(0, 1) > 0.75:
					writeData('test.csv', dataListPmus[x])
				else:
					writeData('train.csv', dataListPmus[x])
					
	'''				

					
def getStabilityLabel(data):
	#print data[0][0], data[1][0]
	if data[-1][0] == 'False':
		#print 'does not converge', data[-1][0]
		return 0

		
	elif int(data[0][0]) > 2370:
		if int(data[1][0]) < 13:
			#print 'converges and meets standard', data[-1][0], int(data[0][0]), int(data[1][0])
			return 1
	else:
		#print 'converges and does not meet standard', data[-1][0], int(data[0][0]), int(data[1][0])
		return 0
	
def parseConvergenceFile(data):
	characters = str(data).strip(']').strip('[')
	#print characters
	listOfData = []
	tempList = []
	tempString = []
	count = 0
	doneFlag = False
	while count != len(characters):
		if characters[count] == "'":
			count += 1 
		elif characters[count] == '[':
			
			count += 1
			
			flag = True
			while flag:
				while characters[count] != ",":
					
					if characters[count] == "]":
						#print 'DONE'
						flag = False
						break
					if characters[count] != "'":
						if characters[count] != " ":
							tempString.append(characters[count])
						 
					count += 1
				
				tempList.append(''.join(tempString))
				tempString = []
			
				count+=1
			listOfData.append(tempList)	
			tempList = []
			#print listOfData
		
		
		else:
			
			#print 'pass'
			
			while characters[count] != ",":
				if characters[count] != "'":
					if characters[count] != " ":
						tempString.append(characters[count])	
				
				count+=1
				if count == len(characters):
					doneFlag = True
					break
			if len(tempString) != 0:
				tempList.append(''.join(tempString))
				tempString = []
				listOfData.append(tempList)	
				tempList = []

		#print listOfData
		
		if doneFlag == True:
			break
		count+=1	
		
	return listOfData
	
	
def createSubSets(totalList, listOneLength, listTwoLength):
	randomizedList = cp.copy(totalList)
	random.shuffle(randomizedList)

	trainingSet = randomizedList[0:18]
	testingSet = randomizedList[18:24]
	
	
	return trainingSet, testingSet
	
def createTrainTestData():
	totalOpPoints = 24
	opPointsList = range(1, totalOpPoints + 1)
	trainingSet, testingSet = createSubSets(opPointsList, 18, 6)
	
	
	mainDataLoc = os.path.join(os.getcwd(),'rawData')
	for x in range(1, 25):
		opPointName = 'newOp' + str(x)
		dataLoc = os.path.join(mainDataLoc,opPointName)
		
		if x in trainingSet:
			stabilityWrapper(os.path.join(dataLoc, 'convergences.csv'), os.path.join(dataLoc, 'savedPmus.csv'), 50, 'train.csv')
		elif x in testingSet:
			stabilityWrapper(os.path.join(dataLoc, 'convergences.csv'), os.path.join(dataLoc, 'savedPmus.csv'), 10, 'test.csv')

		
	
def main():	
	createTrainTestData()

		
		
if __name__ == "__main__": 
	main()	