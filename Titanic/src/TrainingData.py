'''
Created on Apr 8, 2017

@author: dave
'''
import numpy as np

class ProblemData():
    '''
    classdocs
    '''

    def __init__(self, X, yLabel):
        '''
        Constructor
        '''
        self.Xtrain = X
        self.yLabel = yLabel

    def defaultPrepare(self):
        self.shuffle()
        self.splitData()
        self.splitY()
        
    def shuffle(self):
        self.Xtrain = self.Xtrain.iloc[np.random.permutation(len(self.Xtrain))]
        
    def fitPolyNomial(self, degree=2):
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(degree)
        self.Xtrain = self.poly.fit_transform(self.Xtrain)
        
    def addPolyNomial(self, data):
        return self.poly.transform(data)

    def splitData(self):
        from sklearn.model_selection import train_test_split
        [self.Xtrain, self.Xtest] = train_test_split(self.Xtrain, test_size = 0.4)
        [self.Xcv, self.Xtest] = train_test_split(self.Xtest, test_size = 0.5)
        
    def splitY(self):
        self.ytrain = self.Xtrain[self.yLabel]
        del self.Xtrain[self.yLabel]
        
        self.ycv = self.Xcv[self.yLabel]
        del self.Xcv[self.yLabel]
        
        self.ytest = self.Xtest[self.yLabel]
        del self.Xtest[self.yLabel]
        
        
    
