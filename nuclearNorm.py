#Source code written by Carter Lassetter
#Last Modified: 1/18/2016
#Vacuum Cleaning Agents
from __future__ import division
import random
import numpy as np
import math
import copy as cp
import csv
import os, sys

import pywt
import attackEstimator as ae

def getMatrixSize(matrixIn):
	'''Get the dimensions of our input matrix, we will look for illegal matrices
	as well and return  'None' if the input is not a matrix.
	'''

	if type(matrixIn) != type([]):
		return None
	
	
	rowCount = len(matrixIn)
	legalMatrix = True
	if rowCount > 0:
		temp = len(matrixIn[0])
	
	for row in matrixIn:
		if temp != len(row):
			legalMatrix = False
			return None
	columnCount = temp
	
	return (rowCount, columnCount)
	
	
def addMatrices(matrices):
	'''Add matrices together, if they don't have the same dimensions then they will not be added together.
	'''
	
	matrixOneSize = matrices[0].shape
	
	
	for x in range(len(matrices)):
		if matrices[x].shape != matrixOneSize:
			print 'trying to add matrices of different shapes...'
			return None
		   
	
	resultantMatrix = np.zeros(matrixOneSize)	
	for y in range(len(matrices)):
		resultantMatrix += matrices[y]

	return resultantMatrix
	
	
def shrinkageOperatorWrapper(input, tau):
	'''Allows the shrinkageWrapper to be used on matrices'''
	#print tau, 'tau'
	matrixSize = input.shape
	input = input.tolist()
	newMatrix = cp.deepcopy(input)
	
	for x in range(0, matrixSize[0]):
		 for y in range(0, matrixSize[1]):
			newMatrix[x][y] = shrinkageOperator(input[x][y], tau) 
			
	return np.matrix(newMatrix)
	
def shrinkageOperator(inputVariable, tau):	
	'''Performs the shrinkage operator on a given single value'''
	
	if inputVariable == 0:  
		sign = 1.0 
	else:
		sign = float(inputVariable)/np.absolute(inputVariable)
	paramOne = np.absolute(inputVariable) - tau
   
	#print paramOne
	maxVal = max(paramOne, 0)
	
	output = sign*maxVal
	#print output
	
	return output
  


def singularValueDecomp(matrix, fullMatrix):

	U, s, V = np.linalg.svd(matrix, full_matrices=fullMatrix)
	columnsU = U.shape[0]
	columnsV = V.shape[0]

	
	s = s.tolist()
	singularVals = np.zeros((1,columnsU,columnsV))
	singularVals = singularVals[0].tolist()
	
	for x in range(0, columnsU):
		for y in range(0, columnsV):
			if x == y and x < len(s):
				singularVals[x][y] = s[x]
				#print singularVals[x][y],x,y
			else:
				singularVals[x][y] = 0.0 
	

				
	return U, np.matrix(singularVals), V
	
	
def nuclearNormMatrixSeparation(originalMatrix, regularizationTerm, penalty, updateTerm, convergence):
	matrixSize = originalMatrix.shape
	if matrixSize == None:
		print 'input matrix is illegal'
		return 0 
	
	if regularizationTerm == False:
		regularizationTerm = 1.0/(math.sqrt(max(matrixSize)))
	
	
	
	lagrangeMatrix = np.zeros(matrixSize)
	currentMeasurementEstimate = np.zeros(matrixSize)
	currentFalseDataEstimate = np.zeros(matrixSize)
	penaltyTerm = penalty
	updateConstant = updateTerm
	

	
	k = 0 
	
	newMeasurementEstimate = cp.deepcopy(currentMeasurementEstimate)
	newFalseDataEstimate = cp.deepcopy(currentFalseDataEstimate) 
	newMeasurementEstimatePrime = cp.deepcopy(newMeasurementEstimate)
	newFalseDataEstimatePrime = cp.deepcopy(newFalseDataEstimate)
	while True:
		j = 0 
		print j
		firstMeasurementDataEstimate = cp.deepcopy(newMeasurementEstimatePrime)
		firstFalseDataEstimate = cp.deepcopy(newFalseDataEstimatePrime)

		
		
		while True:
		 

			shrinkInputOne = addMatrices([originalMatrix,-1.0*newMeasurementEstimatePrime, np.multiply(1.0/penalty,lagrangeMatrix)])
			svdInput = addMatrices([originalMatrix,-1.0*newFalseDataEstimatePrime, np.multiply(1.0/penalty,lagrangeMatrix)])
			U, s, V = singularValueDecomp(svdInput, True)
			singularShrinkage = shrinkageOperatorWrapper(s, (1.0/penalty))

			
			previousFalseDataEstimatePrime = cp.deepcopy(newFalseDataEstimatePrime)
			previousMeasurementEstimatePrime = cp.deepcopy(newMeasurementEstimatePrime)
			
			newFalseDataEstimatePrime = shrinkageOperatorWrapper(shrinkInputOne, regularizationTerm*(1.0/penalty))
			newMeasurementEstimatePrime = U*singularShrinkage*V

			
			j += 1
			
			maxUpdateDifferenceMeasurement = matrixError(previousMeasurementEstimatePrime, newMeasurementEstimatePrime, True)
			maxUpdateDifferenceFalseData = matrixError(previousFalseDataEstimatePrime, newFalseDataEstimatePrime, False)
			maxUpdateDifference = max(maxUpdateDifferenceMeasurement, maxUpdateDifferenceFalseData)
			print maxUpdateDifference, convergence
			if maxUpdateDifference < convergence or j > 100:
				lastMeasurementEstimateUpdate = cp.deepcopy(newMeasurementEstimatePrime)
				lastFalseDataEstimateUpdate = cp.deepcopy(newFalseDataEstimatePrime)
				print 'j: ', j
				break
			
			

			
		oldLagrangeSaved = cp.deepcopy(lagrangeMatrix)
		lagrangeMatrix = lagrangeMatrix + penalty*(originalMatrix - newMeasurementEstimatePrime - newFalseDataEstimatePrime)
		
		penalty = penalty*updateTerm
		
		k += 1
		
		
		maxUpdateDifferenceMeasurementOuter = matrixError(firstMeasurementDataEstimate, lastMeasurementEstimateUpdate, False)
		maxUpdateDifferenceFalseDataOuter = matrixError(firstFalseDataEstimate, lastFalseDataEstimateUpdate, False)
		maxLagrangeDif = matrixError(oldLagrangeSaved, lagrangeMatrix, False)
		
		
		maxUpdateDifferenceOuter = max(maxUpdateDifferenceMeasurementOuter, maxUpdateDifferenceFalseDataOuter, maxLagrangeDif)
		print 'k: ', k, 'maxDifference: ', maxUpdateDifferenceOuter
		if maxUpdateDifferenceOuter < convergence or k > 100:
			break
			
		
	
	np.savetxt("measurementEstimate.csv", newMeasurementEstimatePrime, delimiter=",")
	np.savetxt("falseDataEstimate.csv", newFalseDataEstimatePrime, delimiter=",")
	
	return newMeasurementEstimatePrime, newFalseDataEstimatePrime
		
		
		
def createAttack(inputMatrix, percentOfTimeAttacked, percentOfPmusAttacked, randomMax):
	columns = inputMatrix.shape[1]
	rows = inputMatrix.shape[0]
	
	amountOfCompPMUS = int((percentOfPmusAttacked/100)*rows)
	print amountOfCompPMUS
	attackMatrix = []
	
	for x in range(0, amountOfCompPMUS):
		pmuList = []
		attackStart = random.randint(0, columns-1)
		attackLength = int((percentOfTimeAttacked/100)*columns)
		#print attackStart, attackLength
		overFlow = max(0, attackStart + attackLength - (columns))
		attackStart -= overFlow 
		attackEnd = attackStart + attackLength
		
		#print attackStart, attackLength
		
		for y in range(0, attackStart):
			pmuList.append(0)
		
		for y in range(attackStart, attackEnd):
			pmuList.append(np.random.uniform(-randomMax, randomMax)) 
		
		for y in range(attackEnd, columns):
			pmuList.append(0)
			
		attackMatrix.append(cp.deepcopy(pmuList))
	
	matrixEntries = len(attackMatrix)
	zeroAttack = [0]*columns 

	for z in range(matrixEntries, rows):
		attackMatrix.append(cp.deepcopy(zeroAttack))
	
	#print attackMatrix
	#print '#############################################################################################################################################'
	random.shuffle(attackMatrix)
	attackLocs = []

	for q in xrange(len(attackMatrix)):
		if max(attackMatrix[q]) > 0:
			attackLocs.append(q)


	print attackLocs

	return np.matrix(attackMatrix), attackLocs
	#print attackMatrix
	
	

def accuracyCheck(estimateMeasurement, estimatedAttack, trueMeasurement, trueAttack):
	attackError = addMatrices([estimatedAttack, -1.0*trueAttack])
	attackError = attackError.tolist()
	size1 = trueAttack.shape
	trueAttack = trueAttack.tolist()
	attackErrorPrct = [[0]*size1[1]]*size1[0]
	attackErrorPercentList = []
	residual = 0

	for x in range(len(attackError)):
		for y in range(len(attackError[0])):
			residual+= attackError[x][y]**2
			try:
				attackPrct = abs(attackError[x][y])/trueAttack[x][y]
				if attackPrct > 50:
					print attackError[x][y], trueAttack[x][y], estimatedAttack[x][y]
				attackErrorPrct[x][y] = attackPrct
				attackErrorPercentList.append(attackPrct)
			except ZeroDivisionError:
				attackPrct = attackError[x][y]
				attackErrorPrct[x][y] = attackPrct
				attackErrorPercentList.append(attackPrct)
	print residual
	print 'Average attack error:', sum(attackErrorPercentList)/len(attackErrorPercentList)
	print 'Max attack error:', max(attackErrorPercentList)
	
	print '##################################################################################'
	
	
	
	
	
	'''
	
	estimateMeasurement = estimateMeasurement.tolist()
	size = getMatrixSize(trueMeasurement)
	errorPercentMatrix = [[0]*size[1]]*size[0]
	errorPercentList = []
	for u in range(len(trueMeasurement)):
		for v in range(len(trueMeasurement[0])):
			error = abs((estimateMeasurement[u][v] - trueMeasurement[u][v])/trueMeasurement[u][v])*100
			errorPercentMatrix[u][v] = error
			errorPercentList.append(error)
			
	print 'Average error:', sum(errorPercentList)/len(errorPercentList)
	print 'Max error:', max(errorPercentList)
			
	'''	   
			
			
			
			
			
def importData(inputData, delete):

	with open(inputData, 'r') as f:
		reader = csv.reader(f)
		pmuMeasurements = list(reader)
	
	if delete:
		del pmuMeasurements[0]
	
	pmuMeasurements = [[float(string) for string in inner] for inner in pmuMeasurements]
  
	return pmuMeasurements

	
def matrixError(matrixInput1, matrixInput2, flag):
	if flag:
		print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
		print matrixInput1
		print matrixInput2
		print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
	differenceMatrix = addMatrices([matrixInput1, -1.0*matrixInput2])
	differenceMatrix = np.absolute(differenceMatrix)
	
	return sum(differenceMatrix.ravel())/len(differenceMatrix.ravel())
	#return np.max(differenceMatrix)
   


def findBadPMUs(dataSet, secondSingValuThresh, badDataThreshold):
	U, s, V = np.linalg.svd(dataSet, full_matrices=True)
	count = 0
	fullSetSingVals = s
	pmuList = range(0, dataSet.shape[0])
	#print pmuList
	corruptPMU = []
	
	for z in range(1, s.shape[0]): 
		if s[1]*secondSingValuThresh < s[z]:
			count += 1
		
	count += 1




	bestNucNorm = float('inf')
	pmuToRemove = None



	print np.linalg.norm(dataSet, ord='nuc', axis=None, keepdims=False) - fullSetSingVals[0]
	


	for z in range(0, 15):
		savedRank = np.linalg.matrix_rank(dataSet, tol=0.1)
		for x in xrange(dataSet.shape[0]):
			
			
			newDataSet = np.delete(dataSet, (x), axis=0)
			#print dataSet.shape, dataSet.mean(axis=0).reshape(1,1200).shape
			newDataSet2 = np.concatenate((newDataSet,dataSet.mean(axis=0).reshape(1,1200)))

			U1, s1, V1 = np.linalg.svd(dataSet, full_matrices=True)
			U, s, V = np.linalg.svd(newDataSet, full_matrices=True)
			U2, s2, V2 = np.linalg.svd(newDataSet2, full_matrices=True)
			
			modNucNorm1 = np.linalg.norm(dataSet, ord='nuc', axis=None, keepdims=False) - s1[0]
			modNucNorm = np.linalg.norm(newDataSet, ord='nuc', axis=None, keepdims=False) - s[0]
			modNucNorm2 = np.linalg.norm(newDataSet2, ord='nuc', axis=None, keepdims=False) - s2[0]

			print x, modNucNorm, modNucNorm2, modNucNorm1
			if modNucNorm < bestNucNorm:
				bestNucNorm = modNucNorm 
				pmuToRemove = x 
				bestNewRank = np.linalg.matrix_rank(newDataSet, tol=0.1)

			'''
			for y in range(1, count):

				''
				if  -((s[y] - fullSetSingVals[y])/fullSetSingVals[y]) > badDataThreshold:
					#print fullSetSingVals[y], s[y], x
					pass

			'''


		
		print bestNucNorm
		print 'Rank: ', bestNewRank, 'PMU removed: ', pmuList[pmuToRemove], pmuToRemove

		
		#pmuList = np.concatenate((pmuList,newrow))
		
		
		if bestNewRank == savedRank:
			return corruptPMU
		


		corruptPMU.append(pmuList[pmuToRemove])
		del pmuList[pmuToRemove]
		dataSet = np.delete(dataSet, (pmuToRemove), axis=0)
	

	return 'never converged'


def findBadPMUs2(dataSet, secondSingValuThresh, badDataThreshold):
	U, s, V = np.linalg.svd(dataSet, full_matrices=True)
	count = 0
	fullSetSingVals = s
	pmuList = range(0, dataSet.shape[0])
	print pmuList
	corruptPMU = []
	
	for z in range(1, s.shape[0]): 
		if s[1]*secondSingValuThresh < s[z]:
			count += 1
		
	count += 1

	
	pmuToRemove = None
	for z in range(0, 15):
		currentRank = np.linalg.matrix_rank(dataSet, tol=0.1)
		bestRank = float('inf')
		for x in xrange(dataSet.shape[0]):
			newDataSet = np.delete(dataSet, (x), axis=0)
			U, s, V = np.linalg.svd(newDataSet, full_matrices=True)
			
			modNucNorm = np.linalg.norm(newDataSet, ord='nuc', axis=None, keepdims=False) - s[0]
			#print s[1], s[2], s[3], s[4], s[5], s[6]
			rank = np.linalg.matrix_rank(newDataSet, tol=0.1)
			#print x, rank
		   
			if rank < bestRank:
				bestRank = rank 
				pmuToRemove = x 




		rank = np.linalg.matrix_rank(dataSet, tol=0.1)
	
		print pmuToRemove
		print 'Rank: ', rank, 'PMU removed: ', pmuList[pmuToRemove]


	
		if currentRank == bestRank:
			return corruptPMU

		corruptPMU.append(pmuList[pmuToRemove])
		del pmuList[pmuToRemove]
		
		print pmuToRemove.mean(axis=1)
		#pmuList = np.concatenate((pmuList,newrow))
		
		dataSet = np.delete(dataSet, (pmuToRemove), axis=0)
		savedRank = rank

	return 'never converged'





def partitionMatrix(inputMatrix,partitionSize):
	totalMatrices = []
	windowSize = (inputMatrix.shape[0], int(inputMatrix.shape[1]/2))
	
	#print windowSize

	totalMatrices.append(inputMatrix[0][0])
	totalMatrices = np.split(inputMatrix, partitionSize, axis=1)



	#print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
	return totalMatrices

	
def cutData(pmuData, numbPmus, voltageOrAngle):
	newData = []
	if voltageOrAngle == 'VOLTAGE':
		startIndex = 0 
		endIndex = int(numbPmus)
	elif voltageOrAngle == 'ANGLE':
		startIndex = int(numbPmus)
		endIndex = int(numbPmus*2)
	else:
		return None
		
	for x in range(0, len(pmuData)):
		tempList = []
		for y in range(startIndex, endIndex):
			tempList.append(pmuData[x][y])
		
		newData.append(cp.copy(tempList))
	
	return newData
def main():	

	dataLocation = os.path.join(os.getcwd(), 'totalData')
	
	
	count = 0 
	falsePositivesTotal = 0
	falseNegativesTotal = 0 
	
	for x in range(1,2):
		opPoint = 'newOp' + str(x)
		opPointData = os.path.join(dataLocation, 'newOp1')
		opPointStableData = os.path.join(opPointData, 'Stable')
		opPointUnstableData = os.path.join(opPointData, 'Unstable')
		
		
		for y in range(0, 50):
			flag = True
			inputFile = os.path.join(opPointStableData, 'caseDataMainGrid' + str(y) + '.csv')
			if not os.path.exists(inputFile):
				inputFile = os.path.join(opPointUnstableData, 'caseDataMainGrid' + str(y) + '.csv')
				if not os.path.exists(inputFile):
					flag = False
			
			if flag:
				pmuData = importData(inputFile, True)
				numberOfPMUs = len(pmuData[0])/2
				
		
		
				
				
				pmuData = cutData(pmuData, numberOfPMUs, 'VOLTAGE')
		
				
				
		
				pmuData = np.matrix(pmuData).transpose()
				
				attackMatrix, attackLocs = createAttack(pmuData, 10, 10, 0.01)
				pmuData = np.matrix(pmuData)

				np.savetxt("actualMeasurement.csv", pmuData, delimiter=",")
				np.savetxt("actualAttack.csv", attackMatrix, delimiter=",")
				attackAndMeasMatrix = addMatrices([pmuData, attackMatrix])

				measurementEstimate, attackEstimate = nuclearNormMatrixSeparation(attackAndMeasMatrix, False, 1, 1, 0.0000001)
				accuracyCheck(measurementEstimate, attackEstimate, pmuData, attackAndMeasMatrix)
				falsePositives, falseNegatives = ae.calcPositives()
				
				falsePositivesTotal += falsePositives
				falseNegativesTotal += falseNegatives
				count += 1
		
	print '10,10,0.01'
	print falsePositivesTotal, falseNegativesTotal, count
if __name__ == "__main__": 
	main()	

