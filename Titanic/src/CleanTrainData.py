'''
Created on Apr 6, 2017

@author: dave
'''

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import CleanData

    # Read Data
    trainPath = Path('C:/Users/dave/Downloads/train.csv')
    train = pd.read_csv(trainPath, index_col=0)
    
    train, cv = train_test_split(train, test_size = 0.2)
    
    # Split data into Cross Validation Train
    X = CleanData(train)

    # Pull out y
    yTrain = train['Survived']
    del train['Survived']
    yCV = cv['Survived']
    del cv['Survived']
    
    # Train Model

    # Cross Validate Model
    
    # Test Model
    
    # Save Test Predictions
    
    