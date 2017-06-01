'''
Created on Apr 8, 2017

@author: dave
'''

from sklearn.linear_model import LogisticRegression
import LearnModel

parmC = [0.001, 0.005, 0.01, 0.05, 1, 10, 25, 50, 100]
max_iter=100000

class LearnLR(LearnModel.LearnModel):
    def trainFit(self):
        self.model = LogisticRegression(solver=self.solver, max_iter=max_iter)
        
        super().trainFit()
     
    def learnAll(self):
        self.learnLrNewtonCg()
        self.learnLrLbfgs()
        self.learnLrLibLinear()
        self.learnLrSag()
    
    def learnLrNewtonCg(self):
        self.solver = 'newton-cg'
  
        self.parameters = {'C':parmC}
        
        super().learnRun();
        
    def learnLrLbfgs(self):
        self.solver = 'lbfgs'
  
        self.parameters = {'C':parmC}
        
        super().learnRun();

    def learnLrLibLinear(self):
        self.solver = 'liblinear'

        self.parameters = {'C':parmC}
        
        super().learnRun();

    def learnLrSag(self):
        self.solver = 'sag'
  
        self.parameters = {'C':parmC}
        
        super().learnRun();
