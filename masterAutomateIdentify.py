# File:"C:\Program Files (x86)\PTI\PSSEXplore33\EXAMPLE\My_py.py", generated on TUE, AUG 11 2015   8:29, release 33.05.02
from __future__ import division
from collections import defaultdict
import os,sys
import multiprocessing 
import extractIslandedInformation as eII

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


def my_function(arg):
	while True:
		if arg == 1:
			print 'ok'
		if arg == 2:
			print 'd'

def main():
	slackBuses = [18, 181, 185, 182, 186]
	zone = 5
	
	
	for x in range(1, 25):
		opPoint = 'newOp' + str(x)
		saveTo = os.path.join(os.getcwd(), opPoint)
		process = multiprocessing.Process(target=eII.analyzeGoodOrBadCases, args=(saveTo,))
		process.start()
	
	
	

	
	
if __name__ == "__main__": 
	main()	



