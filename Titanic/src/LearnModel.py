'''
Created on Apr 8, 2017

@author: dave
'''

class LearnModel():
    cv=3
    num_jobs=3
    results=[]
    models=[]
    tData=None
    pData=None

    def __init__(self, trainData=None, predictData=None):
        '''
        Constructor
        '''
        if trainData != None:
            LearnModel.tData = trainData
            LearnModel.pData = predictData
    
    def addModel(self, model):
        LearnModel.models.append(model)
        
    def addMLP(self):
        import LearnMultiLlayerPerceptron;
        self.addModel(LearnMultiLlayerPerceptron.LearnMLP(LearnModel.tData, LearnModel.pData))

    def addSVC(self):
        import LearnSupportVectorClassifier;
        self.addModel(LearnSupportVectorClassifier.LearnSVC())
        
    def addNN(self):
        import LearnNearestNeighbor;
        self.addModel(LearnNearestNeighbor.LearnNN())
        
    def addLR(self):
        import LearnLogisticRegression;
        self.addModel(LearnLogisticRegression.LearnLR())

    def learnAll(self):
        for model in LearnModel.models:
            model.learnAll()
                   
    def learnRun(self):
        self.trainFit()
        score = self.testScore()
        predict = self.predict()
        print(score)
        self.results.append([score,predict])

    def trainFit(self):
        from sklearn.model_selection import GridSearchCV
        
        #print('Training: ' + self.model)
        
        gs = GridSearchCV(self.model, self.parameters, cv=self.cv, n_jobs=self.num_jobs)
        gs.fit(LearnModel.tData.Xtrain, LearnModel.tData.yTrain)
        
        print("The best parameters are %s with a score of %0.2f" % (gs.best_params_, gs.best_score_))
        print(gs.best_estimator_)
        self.estimator = gs.best_estimator_
        return gs
    
    def testScore(self):
        return(self.estimator.score(LearnModel.tData.Xtest, LearnModel.tData.yTest))
    
    def predict(self):
        #print(self.pData)
        return self.estimator.predict(LearnModel.pData)
    

  
# class LearnAll():
#     '''
#     classdocs
#     '''
#     
#     def __init__(self, data, predictData=None):
#         '''
#         Constructor
#         '''
#         import numpy as np
#         print(predictData)
#         x1 = np.zeros(predictData.shape[0])
#         
#         x2 = learnRun(LearnLrNewtonCg(data, predictData));
#         x1 = np.add(x1,x2)
#         print(x2)
#         
#         x2 = learnRun(LearnLrLbfgs(data, predictData));
#         x1 = np.add(x1,x2)
#         print(x2)
#         
#         x2 = learnRun(LearnLrLibLinear(data, predictData));
#         x1 = np.add(x1,x2)
#         print(x2)
#         
#         x2 = learnRun(LearnLrSag(data, predictData));
#         x1 = np.add(x1,x2)
#         print(x2)
# 
#         print(x1)

