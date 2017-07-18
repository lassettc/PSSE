#Source code written by Carter Lassetter
#Last Modified: 1/18/2016
#Vacuum Cleaning Agents
from __future__ import division
import random
import numpy as np
import math
import copy as cp
import csv
import nuclearNorm as nucNorm



def getAttackedPmus(pmuMatrix, residual):
	'''This will track if we label the PMU as attacked or not.  If the input matrix is the ground truth then
	we use a residual of 0 to ensure that we label every PMU that has anything added to it as attacked.  A 0 means
	the PMU was not attacked, 1 will mean it was attacked.
	'''
	
	pmuList = []
	for x in range(0, len(pmuMatrix)):
		print np.linalg.norm(pmuMatrix[x],ord=2)
		if np.linalg.norm(pmuMatrix[x],ord=2) > residual:
			pmuList.append(1)
		else:
			pmuList.append(0)
   
	print 'end'
		
	return pmuList
	
def accuracyCheck(trueAttack, estimatedAttack):
	falsePositives = 0 #The PMU was estimated to have been attacked, but it wasn't attacked in the ground truth
	falseNegatives = 0 #The PMU was estimated to not have been attacked, but it was attacked in the ground truth
	for x in range(0, len(trueAttack)):
		if trueAttack[x] == 1 and estimatedAttack[x] == 0:
			falseNegatives += 1
		elif trueAttack[x] == 0 and estimatedAttack[x] == 1:
			falsePositives += 1
	
	return falsePositives, falseNegatives
	
	
def calcPositives():
	estimatedAttackFile = 'falseDataEstimate.csv'
	actualAttackFile = 'actualAttack.csv'
	
	
	estimateAttack = nucNorm.importData(estimatedAttackFile, False)
	actualAttack = nucNorm.importData(actualAttackFile, False)
	
	groundTruthAttackedPMUs = getAttackedPmus(actualAttack, 0)
	estimatedAttackedPMUs = getAttackedPmus(estimateAttack, 0.01)
	
	print sum(groundTruthAttackedPMUs)
	falsePositives, falseNegatives = accuracyCheck(groundTruthAttackedPMUs, estimatedAttackedPMUs)
	
	return falsePositives, falseNegatives

	
def main():	
	falsePositives, falseNegatives = calcPositives()
	
	print falsePositives, falseNegatives
if __name__ == "__main__": 
	main()	