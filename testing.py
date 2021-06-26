import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score
import madmom.evaluation.onsets as evaluation

hop = 1024
fs = 44100

TestSpec=np.load('ExampleTestSpec.npy')
TestTarg=np.load('ExampleTestTarg.npy')
#TrainSpec=np.load('ExampleTrainSpec.npy')

AF = np.load('AFs-CEflsa.npy')
PP = np.load('PPParams.npy')
print(np.shape(AF),PP)
PP[0]=[5,0.15,0.0]
PP[1]=[5,0.26,0.0]
PP[2]=[15,0.2,0.1]

#lamb = [0,5,10,15]

#tmax = np.linspace(0,0.3,num=50)
#print(tmax)
Onsets_truth =[]
Onsets_pred=[]
Peaks_pred=[]
Peaks_truth=[]
results = []

#Res = []

time_line = np.arange(0,2000*hop/fs,hop/fs)

def Truth_peaks(Track,hop=hop,fs=fs):
	onsets=[] 
	for i in range(len(Track)):
		if Track[i] == 1:
			onsets=np.append(onsets,i)
	#print(len(onsets))
	if len(onsets) >0: 
		onsets=(onsets*hop)/float(fs)

	return onsets

#print(np.shape(TestTarg[:,1]))
#Peaks_truth = Truth_peaks(TestTarg[:,2])
for j in range(len(AF[0])):
				Peaks_pred.append(utils.meanPPmm(AF[:,j],PP[j,0],PP[j,1],PP[j,2],hop = hop))
				Peaks_truth.append(Truth_peaks(TestTarg[:,j]))
				results.append(evaluation.OnsetEvaluation(Peaks_pred[j],Peaks_truth[j],window=0.025))
#for j in range(len(lamb)):
#        for i in range(len(tmax)):
	            #Peaks_pred =utils.meanPPmm(AF[:,2],lamb[j],tmax[i],0,hop = hop)
	            
	            #results[i,j] = evaluation.OnsetEvaluation(Peaks_pred,Peaks_truth,window=0.025).fmeasure
        #Res.append(results)


#print(results[:,0],results[:,4])

print(results[0],results[1],results[2])

#print(evaluation.OnsetEvaluation(Peaks_pred[0],Peaks_truth[0],window=0.025))

#print(evaluation.OnsetSumEvaluation(results))

#print(evaluation.OnsetMeanEvaluation(results))

#print(len(Peaks_pred[0]))


def onset_function(peaks):
	onset_function = np.zeros(len(time_line))
	index = np.where(np.isin(time_line, peaks))
	for i in index:
	    onset_function[index]= 1
	return onset_function



onset_function_pred = onset_function(Peaks_pred[2])
onset_function_truth = onset_function(Peaks_truth[2])



#plt.plot(tmax,results[:,0],label='lambda=0')
#plt.plot(tmax,results[:,1],label='lambda=5')
#plt.plot(tmax,results[:,2],label='lambda=10')
#plt.plot(tmax,results[:,3],label='lambda=15')

#plt.xlabel('tmin')
#plt.ylabel('medición-F')



#plt.title('contratiempo RNC-AS')
#plt.legend()


plt.subplot(3,1,1)
plt.plot(onset_function_pred)
plt.subplot(3,1,2)
plt.plot(onset_function_truth)
plt.subplot(3,1,3)
plt.plot(TestTarg[:,2])
plt.show()	



   
   
