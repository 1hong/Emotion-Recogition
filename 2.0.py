# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:06:47 2019

@author: 93668
"""
import os
import scipy
import speechpy
#import sklearn
import numpy
import math
mfcc_len=39
def read_wav(filename):
        return scipy.io.wavfile(filename)
def enframe(wavData, frameSize, overlap):
    coeff = 0.97 # 预加重系数
    wlen = len(wavData)
    step = frameSize - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = numpy.zeros((frameNum, frameSize))
    hamwin = numpy.hamming(frameSize)
    for i in range(frameNum):
        singleFrame = wavData[numpy.arange(i * step, min(i * step +frameSize , wlen))]#每200个为一帧，帧移为80
        singleFrame = numpy.append(singleFrame[0], singleFrame[:-1] - coeff * singleFrame[1:]) # 预加重
        frameData[i, :len(singleFrame)] = singleFrame
        frameData[i, :] = hamwin * frameData[i, :] # 加窗
    sg=numpy.arange(wlen)
    for i in range(wlen):
        sg[i]=frameData[i]
    return sg
        
def read():
    #dataset_folder="C:/a学习/casia_wav"
    #class_labels={"neutral":0,"anger":2,"happy":3,"sad":1}
    #for i, directory in enumerate(class_labels):
        #print("started reading folders",directory)
        #os.chdir(directory)
        for filename in os.listdir('c:\\a学习\\casia_wav\\anger'):
            fs, signal=read_wav(filename)
            signal=enframe(signal,200,120)
            MC=speechpy.feature.mfcc(signal,fs,num_cepstral=mfcc_len)
            #x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(dataset_folder,class_labels)
            print (type(MC))
        return MC
    #return numpy.array(x_train),numpy.array(x_test),numpy(y_train),numpy(y_test)
read()
