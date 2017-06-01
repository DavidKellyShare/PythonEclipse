'''
Created on May 30, 2017

@author: dave
'''

from sklearn import neural_network
import LearnModel

activation = ['identity', 'logistic', 'tanh', 'relu']
hiddenLayerSizes = []
alpha = [0.0001, 0.001, 0.01, 0.1]
max_iter=100000

class LearnMLP(LearnModel.LearnModel):
    '''
    classdocs
    '''
    
    def __init__(self, trainData, predictData=None):
        '''
        Constructor
        '''
        n = trainData.Xtrain.shape[1]
        
        # Create Different Networks
        hiddenLayerSizes.append((n*2))
        hiddenLayerSizes.append((n*2,n*2))
        hiddenLayerSizes.append((n*2,n*2,n*2))
        
        #hiddenLayerSizes.append((n*2,n*2,n*2))

        super().__init__(trainData, predictData)
    
    def trainFit(self):
        print("Training MLPClassifier " + self.solver)
        
        self.model = neural_network.MLPClassifier(solver=self.solver, max_iter=max_iter)

        super().trainFit()
        
    def learnAll(self):
        self.learnMlpLbfgs()
        self.learnMlpSgd()
        self.learnMlpAdam()
    
    def learnMlpLbfgs(self):
        self.solver = 'lbfgs'
         
        self.parameters = {'activation':activation, 'alpha':alpha, 'hidden_layer_sizes': hiddenLayerSizes}
        
        super().learnRun();
        
    def learnMlpSgd(self):
        self.solver = 'sgd'
         
        self.parameters = {'activation':activation, 'alpha':alpha, 'hidden_layer_sizes': hiddenLayerSizes}
        
        super().learnRun();
        
    def learnMlpAdam(self):
        self.solver = 'adam'
         
        self.parameters = {'activation':activation, 'alpha':alpha, 'hidden_layer_sizes': hiddenLayerSizes}
        
        super().learnRun();
        
