import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join
import sys
import utilities as util

from model import SoundCNN

arguments = sys.argv
bpm = int(arguments[1])
samplingRate = int(arguments[2])
mypath = str(arguments[3])
iterations = int(arguments[4])
batchSize = int(arguments[5])

classes,trainX,trainYa,valX,valY,testX,testY = util.processAudio(bpm,samplingRate,mypath)

print trainX.shape

def trainNetConv(maxIter):
	maxA = 0
	maxB = 0
	maxC = 0
	Flag = True
	myModel = SoundCNN(classes)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		myIters = 0
		fullTrain  = np.concatenate((trainX,trainYa),axis=1)
		Clone      = np.reshape(trainX,(trainX.shape[0],32,32))
		Clone      = np.flip(Clone,2)
		Clone      = np.reshape(Clone,(trainX.shape[0],1024))
		Clone      = np.concatenate((Clone,trainYa),axis=1)

		Clone2     = np.column_stack((trainX[:,100:],trainX[:,924:]))
		Clone2     = np.concatenate((Clone2,trainYa),axis=1)

		Clone3     = trainX + np.random.normal(0,1,(trainX.shape))*10
		Clone3     = np.concatenate((Clone3,trainYa),axis=1)

		# best in 50 ~ 100 or rang in 50
		Clone4     = np.column_stack((trainX[:,100:150],trainX[:,50:]))
		Clone4     = np.concatenate((Clone4,trainYa),axis=1)

		fullTrain  = np.row_stack((fullTrain,Clone,Clone4))
	
		fullVal    = np.concatenate((valX,valY),axis=1)
		backup     = 0
		while myIters < maxIter:
			perms = np.random.permutation(fullTrain)
			for i in range(perms.shape[0]/batchSize):
				batch = perms[i *batchSize:(i+1) * batchSize,:]
				batchX,batchYa = np.hsplit(batch,[-1])
				batchY = util.oneHotIt(batchYa)
				sess.run(myModel.train_step,feed_dict={myModel.x: batchX, myModel.y_: batchY, myModel.keep_prob: 0.5})

				train_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:batchX, myModel.y_: batchY, myModel.keep_prob: 1.0})
				val_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:valX, myModel.y_: valY, myModel.keep_prob: 1.0})
				test_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:testX, myModel.y_: testY, myModel.keep_prob: 1.0})
				if myIters%100 == 0:
					print("Step %d, Training accuracy: %g"%(myIters, train_accuracy))
					print("Step %d, Validation accuracy: %g"%(myIters, val_accuracy))
					print("Test accuracy: %g"%(test_accuracy))
					print '\033[92m' + str(maxA) + '\033[0m'

				'''
				if test_accuracy >= maxA or ( train_accuracy >= maxB and val_accuracy >= maxC ):
					backup = sess
					if test_accuracy > 0.50 and train_accuracy > 0.6:
						maxA = test_accuracy 
						maxB = train_accuracy
						maxC = val_accuracy

						Flag = False
						print '\033[93m' + 'write - ' + '\033[0m',
						print '\033[91m' + str(maxA) + '\033[0m'
				'''				

				myIters+= 1

		save_path = saver.save(sess, "./model.ckpt")
                
'''
		if Flag:
			save_path = saver.save(sess, "./model.ckpt")
			print '\033[93m' + 'write' + '\033[0m'

		print maxA
		print maxB
		print maxC

		sess = backup 
		test_accuracy = 0
		for i in range(0,5):
			test_accuracy += myModel.accuracy.eval(session=sess,feed_dict={myModel.x:testX, myModel.y_: testY, myModel.keep_prob: 1.0})
		print("Test accuracy: %g"%(test_accuracy/5))
		test_accuracy = 0
		for i in range(0,5):
			test_accuracy += myModel.accuracy.eval(session=sess,feed_dict={myModel.x:testX, myModel.y_: testY, myModel.keep_prob: 1.0})
		print("Test accuracy: %g"%(test_accuracy/5))
		test_accuracy = 0
		for i in range(0,5):
			test_accuracy += myModel.accuracy.eval(session=sess,feed_dict={myModel.x:testX, myModel.y_: testY, myModel.keep_prob: 1.0})
		print("Test accuracy: %g"%(test_accuracy/5))
'''

trainNetConv(iterations)

