'''
Created on Apr 8, 2017

@author: dave
'''

import logging as log

class PolyData():
    '''
    classdocs
    '''

    def __init__(self, trainingData):
        '''
        Constructor
        '''
        self.Xtrain = trainingData.Xtrain
        
        if (hasattr(trainingData, 'Xcv')):
            self.Xcv = trainingData.Xcv

        self.Xtest = trainingData.Xtest

    def fitPolyNomial(self, degree=2):
        if (hasattr(self, 'isFitPolyNomial')):
            log.warn("Training Data Polynomial data already added.")
            return
        
        self.isFitPolyNomial = True
        
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(degree)
        self.Xtrain = self.poly.fit_transform(self.Xtrain)
        
        if (hasattr(self, 'Xcv')):
            self.Xcv = self.poly.transform(self.Xcv)
    
        self.Xtest = self.poly.transform(self.Xtest)
        
    def addPolyNomial(self, data):
        return self.poly.transform(data)

