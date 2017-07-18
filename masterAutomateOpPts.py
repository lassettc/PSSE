# File:"C:\Program Files (x86)\PTI\PSSEXplore33\EXAMPLE\My_py.py", generated on TUE, AUG 11 2015   8:29, release 33.05.02
from __future__ import division
from collections import defaultdict
import os,sys
import multiprocessing 
import MainSimulationPoland as ms

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
	
	saveTo = os.path.join(os.getcwd(), 'output')
	infile = 'newOp1.sav'
	
	for x in range(1, 2):
		opPoint = 'newOp' + str(x)
		saveTo = os.path.join(os.getcwd(), opPoint)
		infile = opPoint + '.sav'
		process = multiprocessing.Process(target=ms.runIslandingSimulationWrapper, args=(zone, slackBuses, saveTo, infile, 1))
		process.start()
	
	
	

	
	
if __name__ == "__main__": 
	main()	



