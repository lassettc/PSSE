import os
import numpy as np
import csv
import copy as cp
import math
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from random import shuffle
import gc
from sklearn import preprocessing
from random import randint
import random
import itertools
import dataLabeling as dl



def extractDataFromCsv(csvData, voltIndices, angleIndices):
	totalDataFeatures = [] 
	totalDataTargets = []
	with open(csvData) as f:
		 content = f.readlines()
	
	for x in range(0, len(content)):
		tempDataList = []
		for y in voltIndices:
			tempDataList.append(content[x].split(',')[y])
		for z in angleIndices:
			tempDataList.append(content[x].split(',')[z])
		
		totalDataTargets.append(int(content[x].strip('\n').split(',')[-1]))
		totalDataFeatures.append(tempDataList)
	
	return totalDataFeatures, totalDataTargets
def findDataIndices(VoltAngleLabels, pmuLocations):
	voltIndices = []
	angleIndices = []
	with open(VoltAngleLabels) as f:
		reader = csv.reader(f)
		dataLabels = next(reader)
	
	
	for x in range(0, len(dataLabels)):	
		dataLabelList = dataLabels[x].split(' ')
		if 'VOLT' in dataLabelList[0]:
			if dataLabelList[1] in pmuLocations:
				voltIndices.append(x)
		elif 'ANGL' in dataLabelList[0]:
			if dataLabelList[1] in pmuLocations:
				angleIndices.append(x)

	return voltIndices, angleIndices
				
def Import_Data(Training_Data, Testing_Data, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations):
	

	Training_Data = np.genfromtxt(Training_Data,delimiter=',')
	Testing_Data = np.genfromtxt(Testing_Data,delimiter=',')
	

	Target_Train = [0]*len(Training_Data)
	Target_Test = [0]*len(Testing_Data)
	Training_Data = Training_Data.tolist()
	Testing_Data = Testing_Data.tolist()

	for x in xrange(0, len(Training_Data)):
		Target_Train[x] = Training_Data[x][-1]
		Training_Data[x].pop()
	for x in xrange(0, len(Testing_Data)):
		Target_Test[x] = Testing_Data[x][-1]
		Testing_Data[x].pop()
	
	#print dataLabels
	
	if onlyMicrogrid == 'yes':
		Training_Data, Testing_Data, dataLabels = microGridPMUs(dataLabels, Training_Data, Testing_Data)

	
	
	
	
	if len(pmuLocations) != 0:
		Training_Data, Testing_Data, dataLabels = preconfiguredPMUlocs(Training_Data, Testing_Data, dataLabels, pmuLocations)
	
	TotalPMUNumberCount = len(Training_Data[0])/2
	#print len(Training_Data[0])
	if randomize == True:
		Training_Data, Testing_Data, dataLabels = pmuSubset(numbPMUs, Training_Data, Testing_Data, dataLabels, TotalPMUNumberCount)

	#print dataLabels		
	#min_max_scaler = preprocessing.MinMaxScaler()
	#Training_Data_Scaled = min_max_scaler.fit_transform(Training_Data)
	#Training_Data_Scaled = Training_Data_Scaled.tolist()

	#print type(Training_Data_Scaled[0])
	#Testing_Data_Scaled = min_max_scaler.fit_transform(Testing_Data)
	#Training_Data_Scaled = preprocessing.scale(Training_Data)
	#Training_Data_Scaled = Training_Data_Scaled.tolist()
	#Testing_Data_Scaled = preprocessing.scale(Testing_Data)
	#Testing_Data_Scaled = Testing_Data_Scaled.tolist()
	#return Training_Data_Scaled, Target_Train, Testing_Data_Scaled, Target_Test
	
	return Training_Data, Target_Train, Testing_Data, Target_Test
	

def preconfiguredPMUlocs(Training_Data, Testing_Data, dataLabels, pmuLocations):
	newTrainingData = []
	newTestingData = []
	newDataLabels = []
	for x in range(0, len(Training_Data)):
		temp1 = []
		for l in range(0, len(dataLabels)):
			if dataLabels[l].split()[1] in pmuLocations:
				temp1.append(Training_Data[x][l])
				if dataLabels[l] not in newDataLabels:
					newDataLabels.append(dataLabels[l])
					
		newTrainingData.append(temp1)
	
	for y in range(0, len(Testing_Data)):
		temp2 = []
		for k in range(0, len(dataLabels)):
			if dataLabels[k].split()[1] in pmuLocations:
				temp2.append(Testing_Data[y][k])
		
		newTestingData.append(temp2)
		
	
	return newTrainingData, newTestingData, newDataLabels
		
	
	
def pmuSubset(numbPMUs, Training_Data, Testing_Data, dataLabels, TotalPMUNumberCount):
	totalPMUsDelete = TotalPMUNumberCount - numbPMUs
	PMUcurrentValue = TotalPMUNumberCount

	for x in range(0, totalPMUsDelete):
		currentRandomVal = randint(0, PMUcurrentValue - 1)
		#print currentRandomVal
		#print PMUcurrentValue
		for y in range(0, len(Training_Data)):
			del Training_Data[y][currentRandomVal + PMUcurrentValue]
			del Training_Data[y][currentRandomVal]
		
		for z in range(0, len(Testing_Data)):
			del Testing_Data[z][currentRandomVal + PMUcurrentValue]
			del Testing_Data[z][currentRandomVal]			
		del dataLabels[currentRandomVal + PMUcurrentValue]
		del dataLabels[currentRandomVal]

		PMUcurrentValue -= 1
	
	
	return Training_Data, Testing_Data, dataLabels
def microGridPMUs(dataLabels, TrainData, TestData):
	tempDecrement = 0
	labelNumbs = []
	
	
	for x in range(0, len(dataLabels)):
		temp = dataLabels[x].split(' ')
		labelNumbs.append(temp[1])
	
	
	
	for y in range(0, len(labelNumbs)):
		if int(labelNumbs[y]) < 300:
			tempIndex = y - tempDecrement
			del dataLabels[tempIndex]
			for a in range(0, len(TrainData)):
				del TrainData[a][tempIndex]
				
			for b in range(0, len(TestData)):
				del TestData[b][tempIndex]
	
			tempDecrement += 1
			

	
	
	
	
	
	return TrainData, TestData, dataLabels
	
def importTotalData(totalData):
	dataTotal = np.genfromtxt(totalData,delimiter=',')
	dataTotal = dataTotal.tolist()
	shuffle(dataTotal)
	halfDataSize = len(dataTotal)/2
	trainSize = int(math.ceil(halfDataSize / 100.0)) * 100 + 100
	
	trainingData = dataTotal[0:trainSize]
	testingData = dataTotal[trainSize:]
	trainTarget = [0]*len(trainingData)
	testTarget = [0]*len(testingData)	
	
	
	np.savetxt("NewTest.csv", testingData, delimiter=",")
	np.savetxt("NewTrain.csv", trainingData, delimiter=",")	
	
	for x in xrange(0, len(trainingData)):
		trainTarget[x] = trainingData[x][-1]
		trainingData[x].pop()
		trainingData[x].pop()
		
	for x in xrange(0, len(testingData)):
		testTarget[x] = testingData[x][-1]
		testingData[x].pop()
		testingData[x].pop()	
	
	
	
	

	
	
	
	return trainingData, trainTarget, testingData, testTarget

	
def SVM_Technique(Training_Data, Target_Train, Testing_Data, Target_Test):



			
	#print CrossTest_Data[0]
	#print CrossTest_Targets[0]
	#print CrossTrain_Data[0]
	#print CrossTrain_Targets[0]

	
	#Test_Our_Data(CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets)
	print '##########################################################################################'
	
	
def Cross_ValidatedErrors(kernel, CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, Error_Vector, Label_Vector, power, C_vals, gamma):
	count = 0
	flag = 0
	if kernel == 'rbf':
		
		for h in gamma:
			for i in C_vals:
				Temp_Label = []
				Temp_Label.append(h)
				Temp_Label.append(i)
				Label_Vector.append(Temp_Label)
						
				notUsed1, notUsed2, notUsed3, notUsed4, Percent_Right = Test_Our_Data(CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, i, 1, kernel, h, flag)
					
				Error_Vector[count] += Percent_Right
				count += 1	
	
	if kernel == 'poly':
		
		for j in power:
			for k in C_vals:
				Temp_Label = []
				Temp_Label.append(j)
				Temp_Label.append(k)
				Label_Vector.append(Temp_Label)
						
				notUsed1, notUsed2, notUsed3, notUsed4, Percent_Right = Test_Our_Data(CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, k, j, kernel, 0, flag)
				
				Error_Vector[count] += Percent_Right
				count += 1	
	
	
	if kernel == 'linear':
		for m in C_vals:
			Temp_Label = []
			Temp_Label.append(m)
			Label_Vector.append(Temp_Label)
			
			notUsed1, notUsed2, notUsed3, notUsed4, Percent_Right = Test_Our_Data(CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, m, 1, kernel, 0, flag)

			Error_Vector[count] += Percent_Right
			count += 1	
			
	return Error_Vector, Label_Vector
	
	
	
	
	
def Test_Our_Data(X_train, X_test, y_train, y_test, C_Value, Power, KERN, Gamma, flag):
	count = 0
	count_Zeros = 0
	count_ZerosTotal = 0
	count_Ones = 0
	count_OnesTotal = 0

	#print len(X_train)
	X_train, y_train = overSampling(X_train, y_train)
	
	
	
	#print len(X_train)
	
	clf = svm.SVC(C= float(C_Value), class_weight=None, coef0=0.0, degree=Power, gamma=float(Gamma), kernel=KERN, max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
	clf.fit(X_train, y_train) 
	this = clf.predict(X_test)
	#print y_test
	#print this
	for x in xrange(0, len(y_test)):

		if this[x] == y_test[x]:
			count += 1

	#print 'Total correct: %d' %count 
	#print 'Total tested on: %d' %len(y_test)
	
	for z in xrange(0, len(y_test)):
		if y_test[z] == 0:
			count_ZerosTotal +=1
			if this[z] == y_test[z]:
				count_Zeros += 1
			else:
				pass
				#print 'zero'
				#print z
		if y_test[z] == 1:
			count_OnesTotal += 1
			if this[z] == y_test[z]:
				count_Ones += 1
			else:
				pass
				#print 'one'
				#print z	
				
				
	if flag == 1:
		print 'Total correct: %d' %count 
		print 'Total tested on: %d' %len(y_test)
						
		print 'Total of class one correct: %d' %count_Ones
		print 'Total of class one: %d' %count_OnesTotal
		print 'Total of class zero correct: %d' %count_Zeros
		print 'Total of class zero: %d' %count_ZerosTotal
	
	return float(count_Ones), float(count_OnesTotal), float(count_Zeros), float(count_ZerosTotal), float(count)/float(len(y_test))
	

def separateClassExamples(featureList, targetList):
	zeroFeatures = []
	oneFeatures = []
	zeroTargets = []
	oneTargets = []
	for x in range(0, len(featureList)):
		if targetList[x] == 0:
			zeroFeatures.append(featureList[x])
			zeroTargets.append(targetList[x])
		elif targetList[x] == 1:
			oneFeatures.append(featureList[x])
			oneTargets.append(targetList[x])
	return zeroFeatures, oneFeatures, zeroTargets, oneTargets
	

def overSamplingHelper(minorityClassFeatures, minorityClassTargets, targetAmount):
	newSampledMinorityClassFeatures = cp.copy(minorityClassFeatures)
	newSampledMinorityClassTargets = cp.copy(minorityClassTargets)
	
	featureCopy = cp.copy(minorityClassFeatures)
	targetCopy = cp.copy(minorityClassTargets)
	
	for x in range(0, targetAmount - len(minorityClassFeatures)):
		randomInt = randint(0, len(featureCopy) - 1)
		newSampledMinorityClassFeatures.append(featureCopy[randomInt])
		newSampledMinorityClassTargets.append(targetCopy[randomInt])
		
		'''
		del featureCopy[randomInt]
		del targetCopy[randomInt]
		
		
		if len(featureCopy) == 0:
			featureCopy = cp.copy(minorityClassFeatures)
			targetCopy = cp.copy(minorityClassTargets)
		'''
		
	return newSampledMinorityClassFeatures, newSampledMinorityClassTargets
	
def overSampling(featureList, targetList):
	zeroFeatures, oneFeatures, zeroTargets, oneTargets = separateClassExamples(featureList, targetList)
	numberZeros = len(zeroFeatures)
	numberOnes = len(oneFeatures)
	if numberZeros > numberOnes:
		newOneFeatures, newOneTargets = overSamplingHelper(oneFeatures, oneTargets, numberZeros)
		totalDataFeatures = newOneFeatures + zeroFeatures
		totalDataTargets = newOneTargets + zeroTargets
	elif numberOnes > numberZeros:
		newZeroFeatures, newZeroTargets = overSamplingHelper(zeroFeatures, zeroTargets, numberOnes)
		totalDataFeatures = newZeroFeatures + oneFeatures
		totalDataTargets = newZeroTargets + oneTargets	
	
	else:
		#print numberZeros, numberOnes
		totalDataFeatures = zeroFeatures + oneFeatures
		totalDataTargets = zeroTargets + oneTargets		

		
	#print len(zeroFeatures), len(oneFeatures), len(zeroTargets), len(oneTargets)
	#print len(totalDataFeatures), len(totalDataTargets)
	
	return totalDataFeatures, totalDataTargets


def breakDataIntoTrainValidationSets(data, sets):
	dataSets = []
	for x in range(0, sets):
		dataSets.append([])
	
	while len(data) != 0:
		for y in range(0, sets):
			dataSets[y].append(data.pop())
			if len(data) == 0:
				break
	
	return dataSets
	
def EightFold_CrossValidation(Training_Data, Target_Train, fold, subset):	

	C_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10]
	Kernels = ['rbf']
	power = [1, 2, 3]
	gamma = [0.0001, 0.001, 0.01, 0, 0.00001, 0.000001]
	rbf_ErrorVector = [0]*len(C_vals)*len(gamma)
	linear_ErrorVector = [0]*len(C_vals)
	poly_ErrorVector = [0]*len(C_vals)*len(power)	

	TrainingData = cp.deepcopy(Training_Data)
	
	
	for j in range(0, len(Training_Data)):
		TrainingData[j].append(Target_Train[j])
	
	
	
	rbfErrors = [0]*len(rbf_ErrorVector)
	polyErrors = [0]*len(poly_ErrorVector)
	linearErrors = [0]*len(linear_ErrorVector)
	
	
	
	shuffle(TrainingData)
	
	brokenUpData = breakDataIntoTrainValidationSets(TrainingData, fold)
	

	
	



	for x in range(0, fold):
		trainingDataSubGroup = []
		trainingDataSubGroupList = cp.deepcopy(brokenUpData)
		testingDataSubGroup = trainingDataSubGroupList[x]
		del trainingDataSubGroupList[x]
		
		for k in range(0, len(trainingDataSubGroupList)):
			trainingDataSubGroup += trainingDataSubGroupList[k]
		
		
		
		CrossTestTargets = []
		CrossTrainTargets = []
		
		for b in range(0, len(trainingDataSubGroup)):
			CrossTrainTargets.append(trainingDataSubGroup[b][-1])
			trainingDataSubGroup[b].pop()
		
		for c in range(0, len(testingDataSubGroup)):
			CrossTestTargets.append(testingDataSubGroup[c][-1])
			testingDataSubGroup[c].pop()			
		
		
		print len(CrossTrainTargets), len(trainingDataSubGroup), len(CrossTestTargets), len(testingDataSubGroup)
		
		
		CrossTest_Data = cp.deepcopy(testingDataSubGroup)
		CrossTest_Targets = cp.deepcopy(CrossTestTargets)
		CrossTrain_Data = cp.deepcopy(trainingDataSubGroup)
		CrossTrain_Targets = cp.deepcopy(CrossTrainTargets)
		rbf_LabelVector =[]
		linear_LabelVector =[]
		poly_LabelVector = []

			
	
		CrossTrain_Data, CrossTrain_Targets = overSampling(CrossTrain_Data, CrossTrain_Targets)
		CrossTest_Data, CrossTest_Targets = overSampling(CrossTest_Data, CrossTest_Targets)
		print len(CrossTest_Data), len(CrossTest_Targets)
		print len(CrossTrain_Data), len(CrossTrain_Targets)
		
		
		for g in Kernels:
			if g == 'rbf':
				rbf_ErrorVector, RBF_LABELS = Cross_ValidatedErrors('rbf', CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, rbf_ErrorVector, rbf_LabelVector, power, C_vals, gamma)
			elif g == 'poly':
				poly_ErrorVector, POLY_LABELS = Cross_ValidatedErrors('poly', CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, poly_ErrorVector, poly_LabelVector, power, C_vals, gamma)
			elif g == 'linear':
				linear_ErrorVector, LINEAR_LABELS = Cross_ValidatedErrors('linear', CrossTrain_Data, CrossTest_Data, CrossTrain_Targets, CrossTest_Targets, linear_ErrorVector, linear_LabelVector, power, C_vals, gamma)

			
		print 'pass %d done' %x
			#del CrossTest_Data
			#del 
			
			
	for s in range(0, len(rbf_ErrorVector)):
		rbfErrors[s] += rbf_ErrorVector[s]
		
		
	for d in range(0, len(poly_ErrorVector)):
		polyErrors[d] += poly_ErrorVector[d]
		
	for q in range(0, len(linear_ErrorVector)):
		linearErrors[q] += linear_ErrorVector[q]
		
		
		
		
	maxRBF = max(rbfErrors)
	maxRBFlocation = []
	maxPOLY = max(polyErrors)
	maxPOLYlocation = []
	maxLINEAR = max(linearErrors)
	maxLINEARlocation = []
	
	for t in range(0, len(rbfErrors)):
		if rbfErrors[t] == maxRBF:
			maxRBFlocation.append(t)
			
	for v in range(0, len(polyErrors)):		
		if polyErrors[v] == maxPOLY:
			maxPOLYlocation.append(v)
			
	for u in range(0, len(linearErrors)):
		if linearErrors[u] == maxLINEAR:
			maxLINEARlocation.append(u)




	totalRBFLabels = []
	for q in maxRBFlocation:
		print RBF_LABELS[q]
		totalRBFLabels.append(RBF_LABELS[q])
	print maxRBF
	'''
	for r in maxPOLYlocation:
		print POLY_LABELS[r]
	print maxPOLY
	for s in maxLINEARlocation:
		print LINEAR_LABELS[s]
	print maxLINEAR
	'''
	return totalRBFLabels, maxRBF

	
def returnRandomSubset(pmuLocations, numbPMUs):
	pmusToRemove = len(pmuLocations) - numbPMUs
	for x in range(0, pmusToRemove):
		random.shuffle(pmuLocations)
		pmuLocations.pop()
	



	return pmuLocations

def mostOccuringElementFromList(list):
	kernelDict = {}
	for z in range(0, len(list)):
		kernelDict[str(list[z])] =  list.count(list[z])

	maxOccurances = kernelDict[max(kernelDict, key=kernelDict.get)]
	count = 0
	for key in kernelDict:
		if kernelDict[key] == maxOccurances:
			count+= 1 
	if count > 1:
		return "['NULL', 'NULL']", True

	return max(kernelDict, key=kernelDict.get), False
		
		
		
		
		

def writeFile(fileName, separateText):
	with open(fileName, "a") as myfile:
		for x in range(0, len(separateText)):
			myfile.write(separateText[x])
	
	
	
def crossValidationWrapper(amountOfTimesToRun, fold, trainingData, testingData, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations, list, printOut):

	bestKernels = list
	for x in range(0, amountOfTimesToRun):
		Training_Data, Target_Train, Testing_Data, Target_Test = Import_Data(trainingData, testingData, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations)	
		subset = int(float(len(Training_Data))/10)
		RBF_Kernels, Accuracy = EightFold_CrossValidation(Training_Data, Target_Train, fold, subset)        
		for y in range(0, len(RBF_Kernels)):
			bestKernels.append(RBF_Kernels[y]) 
    
	
	bestHyperParams, moreThanOne = mostOccuringElementFromList(bestKernels)
	if moreThanOne:
		print 'Running one more to get best params'
		bestHyperParams = crossValidationWrapper(1, 10, trainingData, testingData, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations, bestKernels, False)
	
	
	if type(bestHyperParams) == type([]):
		pass
	elif type(bestHyperParams) == type('a'):
		bestHyperParams = bestHyperParams.strip(']').strip('[').split(',')
		
	if printOut:	
			
		print bestHyperParams[0], bestHyperParams[1]
		print bestKernels.count([0.01, 1]), '0.01, 1'
		print bestKernels.count([0.01, 10]), '0.01, 10' 
		print bestKernels.count([0.01, 0.1]), '0.01, 0.1' 
		print bestKernels.count([0.01, 0.01]), '0.01, 0.01' 	
		print bestKernels.count([0.01, 0.001]), '0.01, 0.001' 		
		print bestKernels.count([0.001, 1]), '0.001, 1'
		print bestKernels.count([0.001, 10]), '0.001, 10' 
		print bestKernels.count([0.001, 0.1]), '0.001, 0.1' 
		print bestKernels.count([0.001, 0.01]), '0.001, 0.01' 	
		print bestKernels.count([0.001, 0.001]), '0.001, 0.001' 		
		print bestKernels.count([0.0001, 1]), '0.0001, 1'
		print bestKernels.count([0.0001, 10]), '0.0001, 10' 
		print bestKernels.count([0.0001, 0.1]), '0.0001, 0.1' 
		print bestKernels.count([0.0001, 0.01]), '0.0001, 0.01' 	
		print bestKernels.count([0.0001, 0.001]), '0.0001, 0.001' 		
		print bestKernels.count([0.00001, 1]), '0.00001, 1'
		print bestKernels.count([0.00001, 10]), '0.00001, 10' 
		print bestKernels.count([0.00001, 0.1]), '0.00001, 0.1' 
		print bestKernels.count([0.00001, 0.01]), '0.00001, 0.01' 	
		print bestKernels.count([0.00001, 0.001]), '0.00001, 0.001' 	

	
	return bestHyperParams
	
	
def createRandomPMUsets(totalPMUs, numbPMUs, randomSetNum):
	setList = []
	for x in range(0, randomSetNum):
		totalPMUsSet = cp.copy(totalPMUs)
		pmuLocations = returnRandomSubset(totalPMUsSet, numbPMUs)
		setList.append(pmuLocations)
	
	return setList
	
	
	
def testOurClassifierWrapper(saveToResultsFile, trainingData, testingData, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations, flag, cVal, betaVal):
	onesRightTotal = 0
	onesCountTotal = 0
	zerosRightTotal = 0 
	zerosCountTotal = 0
	
	
	timesToRun = 20
	for x in range(0, timesToRun):
		Training_Data, Target_Train, Testing_Data, Target_Test = Import_Data(trainingData, testingData, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations)
		fold = 10	
		subset = int(float(len(Training_Data))/10)
		
		onesCount, onesTotal, zerosCount, zerosTotal, totalAccuracy = Test_Our_Data(Training_Data, Testing_Data, Target_Train, Target_Test, cVal, 1, 'rbf', betaVal, flag)
		onesRightTotal += onesCount
		onesCountTotal += onesTotal
		zerosRightTotal += zerosCount 
		zerosCountTotal += zerosTotal
	

	separateText = ['\nTotal averaged over 20 attempts \n', 'Total of class one correct: ', str((onesRightTotal/timesToRun)), '\nTotal of class one: ', 
		str((onesCountTotal/timesToRun)), '\nTotal of class zero correct: ', str((zerosRightTotal/timesToRun)), ' \nTotal of class zero: ', str((zerosCountTotal/timesToRun)), "\n"]
	writeFile(saveToResultsFile, separateText)	
	
	
	print 'Total averaged over 20 attempts'	
	print 'Total of class one correct: %f' %(onesRightTotal/timesToRun)
	print 'Total of class one: %f' %(onesCountTotal/timesToRun)
	print 'Total of class zero correct: %f' %(zerosRightTotal/timesToRun)
	print 'Total of class zero: %f' %(zerosCountTotal/timesToRun)
		
		
	return (onesRightTotal/timesToRun), (onesCountTotal/timesToRun), (zerosRightTotal/timesToRun), (zerosCountTotal/timesToRun)
def main():
	
	subset = 10
	fold = 10
	subSamplingFlag = 1
	Subset_Size = 60
	Xfold = 10
	flag = 1
	#numbPMUs = 9
	onlyMicrogrid = 'no'
	onesCountTotal = 0
	onesTotalTotal = 0 
	zerosCountTotal = 0
	zerosTotalTotal = 0
	flag = 1
	onlyMicrogrid = 'no'
	randomize = False
	pmuLocations = ['10', '15', '118', '125', '127', '139', '140', '141', '214', '225', '303', '315', '335', '1607', '1761', '171', '165', '126', '186', '128', '166', '174', '167', '178', '2218', '2124', '2249', '2331', '2226', '2234']
	
	pmuLocations = createRandomPMUsets(pmuLocations, 10, 1)
	pmuLocations = pmuLocations[0]
	
	VoltAngleLabels = 'pmuBusLabels.csv'
	trainingData = 'train.csv'
	testingData = 'test.csv'
	totalData = 'Total_Data.csv'
	numbPMUs = len(pmuLocations)
	pmuLocations = returnRandomSubset(pmuLocations, numbPMUs)
	print set(pmuLocations)
	
	
	
	voltIndices, angleIndices = findDataIndices(VoltAngleLabels, pmuLocations)
	Training_Data, Target_Train = extractDataFromCsv(trainingData, voltIndices, angleIndices)
	Testing_Data, Target_Test = extractDataFromCsv(testingData, voltIndices, angleIndices)
	#print Training_Data[0][-1]
	#print Training_Data[1][-1]
	
	#Training_Data, Target_Train, Testing_Data, Target_Test = Import_Data(trainingData, testingData, VoltAngleLabels, numbPMUs, onlyMicrogrid, randomize, pmuLocations)
	#print len(Training_Data[0])
	#RBF_Kernels, Accuracy = EightFold_CrossValidation(Training_Data, Target_Train, fold, subset)
	
	#onesCount, onesTotal, zerosCount, zerosTotal, totalAccuracy = Test_Our_Data(Training_Data, Testing_Data, Target_Train, Target_Test, 10, 1, 'rbf', 0.0001, flag)
	
		
if __name__ == "__main__":
    main()	



	
	
	
	