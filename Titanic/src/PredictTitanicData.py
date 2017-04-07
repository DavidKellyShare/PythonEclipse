'''
Created on Apr 6, 2017

@author: dave
'''

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import CleanTitanicData

    # Read Data
    trainPath = Path('../Data/train.csv')
    train = pd.read_csv(trainPath, index_col=0)
    
    # Split data into Cross Validation Train
    [train, cv] = train_test_split(train, test_size = 0.2)
    
    # Pull out y
    yTrain = train['Survived']
    del train['Survived']
    yCV = cv['Survived']
    del cv['Survived']
    
    # Clean Features
    #
    CleanTitanicData.CleanData(train)
    CleanTitanicData.CleanData(cv)
    
    # Scale Features
    #
    import ScaleData
    titanicScaler = ScaleData.ScaleData()
    trainScaled = titanicScaler.fitScaleData(train)
    cvScaled = titanicScaler.scaleData(cv)
    # pd.DataFrame(scaler.fit_transform(data), columns=data.columns).as_matrix()

    # Train Model
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(trainScaled, yTrain)
    print(clf)
    
    # Cross Validate Model
    print(clf.predict(cvScaled))
    print(yCV.values)
    
        
    # Test Model
    
    # Save Test Predictions
    
    