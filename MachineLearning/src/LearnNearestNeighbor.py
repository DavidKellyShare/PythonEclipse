'''
Created on Apr 8, 2017

@author: dave
'''

from sklearn import neighbors
import LearnModel

n_neighbors = [1,5,10,20,40,80]
weights = ['uniform', 'distance']
leaf_size = [30,50,100]


def learnRun(learn):
    learn.trainFit()
    check = learn.testScore()
    print(check)
    return learn.predict()

class LearnNN(LearnModel.LearnModel):
    
    def trainFit(self):
        self.model = neighbors.KNeighborsClassifier(algorithm=self.algorithm)

        super().trainFit()
         
    def learnAll(self):
        self.learnNnBallTree();
        self.learnNnKdTree();
        self.learnNnBrute();

    
    def learnNnBallTree(self):
        self.algorithm = 'ball_tree'
  
        self.parameters = {'n_neighbors':n_neighbors,'weights':weights, 'leaf_size':leaf_size}
        
        super().learnRun()
        
    def learnNnKdTree(self):
        self.algorithm = 'kd_tree'
  
        self.parameters = {'n_neighbors':n_neighbors,'weights':weights, 'leaf_size':leaf_size}
        
        super().learnRun()
        
    def learnNnBrute(self):
        self.algorithm = 'brute'
  
        self.parameters = {'n_neighbors':n_neighbors,'weights':weights, 'leaf_size':leaf_size}
        
        super().learnRun()
        
