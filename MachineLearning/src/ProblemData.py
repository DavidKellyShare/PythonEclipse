'''
Created on Apr 8, 2017

@author: dave
'''
import numpy as np
import logging as log

class ProblemData():
    '''
    classdocs
    '''

    def __init__(self, data, yLabel, createCV=False):
        '''
        Constructor
        '''
        self.Xtrain = data
        self.yLabel = yLabel
        self.createCV=createCV

    def defaultPrepare(self):
        self.shuffle()
        self.splitData()
        self.splitY()
        self.scaleFitData()
        
    def shuffle(self):
        if (hasattr(self, 'isDataSplit')):
            log.warn("Training Data can not be shuffled after split.")
            return

        if (hasattr(self, 'isDdataSplitY')):
            log.warn("Training Data can not be shuffled after Y split.")
            return
            
        self.Xtrain = self.Xtrain.iloc[np.random.permutation(len(self.Xtrain))]
        
    def splitData(self):
        if (hasattr(self, 'isDataSplit')):
            log.warn("Training Data has already been split into CV and TEST.")
            return
        
        self.isDataSplit = True
        
        if (self.createCV):
            test_size = 0.4
        else:
            test_size = 0.3

        from sklearn.model_selection import train_test_split
        [self.Xtrain, self.Xtest] = train_test_split(self.Xtrain, test_size = test_size)
        
        if (self.createCV):
            [self.Xcv, self.Xtest] = train_test_split(self.Xtest, test_size = 0.5)
        
    def splitY(self):
        if (hasattr(self, 'isDataSplitY')):
            log.warn("Result Data Y has already been split from training Data.")
            return
        
        self.isDataSplitY = True
        
        self.yTrain = self.Xtrain[self.yLabel]
        del self.Xtrain[self.yLabel]
        
        if (hasattr(self, 'Xcv')):
            self.ycv = self.Xcv[self.yLabel]
            del self.Xcv[self.yLabel]
        
        self.yTest = self.Xtest[self.yLabel]
        del self.Xtest[self.yLabel]
        
    def scaleFitData(self):
        if (not hasattr(self, 'isDataSplitY')):
            log.warn("Training Data cannot be scaled until after Y is split.")
            return

        if (hasattr(self, 'isDataScaled')):
            log.warn("Training Data has already been scaled.")
            return
        
        self.isDataScaled = True
        
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(copy=False)
        self.Xtrain = self.scaler.fit_transform(self.Xtrain)
        
        if (hasattr(self, 'Xcv')):
            self.Xcv = self.scaleTransformData(self.Xcv)
        
        self.Xtest = self.scaleTransformData(self.Xtest)
        
    def scaleTransformData(self, data):
        return self.scaler.transform(data)
