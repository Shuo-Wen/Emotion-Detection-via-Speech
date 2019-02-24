import cv2
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join

def oneHotIt(Y):
	m = Y.shape[0]
	Y = Y[:,0]
	OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
	OHX = np.array(OHX.todense()).T
	return OHX

def DataNormalization(X):
	X = X.transpose()
	X = (X - np.mean(X,axis=0))/X.max(axis=0)
	X = X.transpose()
 
	return X

def processAudio(bpm,samplingRate,mypath):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	classes = len(onlyfiles)
	
	dataList = []
	labelList = []
	for ix,audioFile in enumerate(onlyfiles):
		rate, audData = scipy.io.wavfile.read(mypath+audioFile)
		seconds = audData.shape[0]/samplingRate
		samples = seconds * bpm / 60
		audData = np.reshape(audData[0:samples*((seconds*samplingRate)/samples)],[samples,(seconds*samplingRate)/samples])
		#print audData.shape, seconds, samples
		for data in audData:
			dataList.append(data)
		labelList.append(np.ones([samples,1])*ix)

	Ys = np.concatenate(labelList)

	specX = np.zeros([len(dataList),1024])
	xindex = 0
	for x in dataList:
		work = matplotlib.mlab.specgram(x)[0]
		worka = work[0:60,:]
		worka = scipy.misc.imresize(worka,[32,32])
		worka = np.reshape(worka,[1,1024])
		specX[xindex,:] = worka
		xindex +=1

	#specX = DataNormalization(specX)
	split1 = specX.shape[0] - specX.shape[0]/20 
	split2 = (specX.shape[0] - split1) / 2

	formatToUse = specX
	Data = np.concatenate((formatToUse,Ys),axis=1)
	DataShuffled = np.random.permutation(Data)
	newX,newY = np.hsplit(DataShuffled,[-1])
	trainX,otherX = np.split(newX,[split1])
	trainYa,otherY = np.split(newY,[split1])
	valX, testX = np.split(otherX,[split2])
	valYa,testYa = np.split(otherY,[split2])
	trainY = oneHotIt(trainYa)
	testY = oneHotIt(testYa)
	valY = oneHotIt(valYa)
	return classes,trainX,trainYa,valX,valY,testX,testY
