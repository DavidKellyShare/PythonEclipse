'''
Created on Apr 8, 2017

@author: dave
'''
import pandas as pd
import logging as log
import LearnModel



class TitanicData():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def learnData(self):
        
        models = LearnModel.LearnModel(self.data, self.testData)
        models.addSVC()
        models.addLR()
        models.addNN()
        models.addMLP()
        models.learnAll()
        
        #import LearnMultiLlayerPerceptron;
        #LearnMultiLlayerPerceptron.LearnMLP(self.data, self.testData).learnAll()

        #import LearnSupportVectorClassifier;
        #LearnSupportVectorClassifier.LearnSVC(self.data, self.testData).learnAll()

        #import LearnNearestNeighbor;
        #LearnNearestNeighbor.LearnNN(self.data, self.testData).learnAll()
        
        #import LearnLogisticRegression;
        #LearnLogisticRegression.LearnLR(self.data, self.testData).learnAll()

        bestResult = 0
        for result in LearnModel.LearnModel.results:
            if result[0] > bestResult:
                bestResult = result[0]
                bestPredict = result[1]
        print(bestResult)
        print(bestPredict)
        
        #print(self.resultFrame)
        result = pd.DataFrame(index=self.ids, data=bestPredict, columns=['Survived'])
        #self.resultFrame['Survived'] = pd.Series(bestPredict)
        print(result)
        result.to_csv('../Data/answer.csv', index_label='PassengerId')
        
        #np.savetxt("../Data/answer.csv", bestPredict, fmt='%1x', delimiter=",")

        #x1 = np.zeros(predictData.shape[0])
        
        #x2 = learnRun(LearnLrNewtonCg(data, predictData));
        #x1 = np.add(x1,x2)

        #print(LearnSupportVectorClassifier.p)
        
        # Plot Scores
        #svc.setrbf()
        #print(len(self.data.Xtrain), len(self.data.yTrain))
        #print(len(self.data.Xcv), len(self.data.ycv))
        #svc.plotScores(self.data.Xtrain, self.data.yTrain, self.data.Xcv, self.data.ycv)

    def readTrainData(self, fileName='../Data/train.csv'):
        from pathlib import Path 
        import ProblemData
        
        # Read Data
        trainPath = Path(fileName)
        df = pd.read_csv(trainPath, index_col=0)
        log.info("Reading Titanic Training Data")
        self.CleanData(df)
        #print(df[['CabinNum','Pclass']])
        #print(df)
        #return
        self.data = ProblemData.ProblemData(df, 'Survived')
        self.data.defaultPrepare()
        

    def readTestData(self, fileName='../Data/test.csv'):
        from pathlib import Path 

        # Read Data
        testPath = Path(fileName)
        self.testData = pd.read_csv(testPath, index_col=0)
        self.ids = self.testData.index.values
        log.info("Reading Titanic Test Data")
        self.CleanData(self.testData)
        self.testData = self.data.scaleTransformData(self.testData)

        #print(np.unique(self.testData))
        #print(self.testData.columns)
        #for col in self.testData.columns:
        #    print(col, self.testData[col].unique())

    def CleanData(self, data):
        # Clean Cabin
        CabinId={'A':1,'B':2,'C':3,'D':4,'E':5, '2':0}
        data['CabinType'] = data['Cabin'].str.extract('^([ABCDE]).*', expand=False).apply(lambda x: CabinId.get(x,6))
        #data['CabinNum'] = pd.to_numeric(data['Cabin'].str.extract('([0123456789]+)', expand=False).fillna(0))
        del data['Cabin']
        
        # Clean Embarked
        Embark = {'S':1,'C':2,'Q':3}
        data['EmbarkId'] = data['Embarked'].apply(lambda x: Embark.get(x,0))
        #data['Embarked'].unique()
        del data['Embarked']
        
        # Clean Fare
        #data['Fare'].unique()
        del data['Fare']
        
        # Clean Ticket
        del data['Ticket']
        
        # Clean Age
        #data['Age'].unique()
        #data[data['Age'].isnull()]
        #
        # Save mean for subsequent calls
        #
        data['Age'].fillna(data['Age'].mean(), inplace=True)
        
        # Clean Sex
        #data['Sex'].unique()
        Sex = {'male':1,'female':2}
        data['SexId'] = data['Sex'].apply(lambda x: Sex.get(x,0))
        del data['Sex']
        
        # Clean Name
        #data['Name'].unique()
        del data['Name']
        
        return data

if __name__ == '__main__':
    titanicData = TitanicData()
    
    titanicData.readTrainData()
    titanicData.readTestData()
    
    titanicData.learnData()
        
    

