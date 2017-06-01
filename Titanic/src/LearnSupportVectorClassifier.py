'''
Created on Apr 8, 2017

@author: dave
'''

from sklearn import svm
import LearnModel

parmC = [1,10,25,50,100]
parmGamma = [0.0001, 0.001, 0.01, 0.1]
cache_size=10000
max_iter=100000
degree = [3,5,7]

class LearnSVC(LearnModel.LearnModel):
    
    def trainFit(self):
        self.model = svm.SVC(kernel=self.kernel, cache_size=cache_size, max_iter=max_iter, decision_function_shape='ovr')

        super().trainFit()
    
    def learnAll(self):
        self.learnSvcRbf()
        
        self.learnSvcLinear()
        
        self.learnSvcPoly()
        
        self.learnSvcSigmoid()
    
    def learnSvcRbf(self):
        self.kernel = 'rbf'
  
        self.parameters = {'C':parmC, 'gamma':parmGamma}
        
        super().learnRun();
        
    def learnSvcLinear(self):
        self.kernel = 'linear'
  
        self.parameters = {'C':parmC}
        
        super().learnRun();
        
    def learnSvcPoly(self):
        self.kernel = 'poly'
        
        self.parameters = {'C':parmC, 'gamma':parmGamma, 'degree':degree}
  
        super().learnRun();
        
    def learnSvcSigmoid(self):
        self.kernel = 'sigmoid'
  
        self.parameters = {'C':parmC, 'gamma':parmGamma}
        
        super().learnRun();
