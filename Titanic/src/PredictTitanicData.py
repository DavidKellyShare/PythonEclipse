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
    Xtrain = pd.read_csv(trainPath, index_col=0)
    
    # Split data into Cross Validation Train
    [Xtrain, Xcv] = train_test_split(Xtrain, test_size = 0.2)
    
    # Pull out y
    yTrain = Xtrain['Survived']
    del Xtrain['Survived']
    ycv = Xcv['Survived']
    del Xcv['Survived']
    
    # Clean Features
    #
    CleanTitanicData.CleanData(Xtrain)
    CleanTitanicData.CleanData(Xcv)
    
    # Add Polynomial Features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(4)
    Xtrain = poly.fit_transform(Xtrain)
    Xcv = poly.transform(Xcv)
    
    # Scale Features
    #
    import ScaleData
    titanicScaler = ScaleData.ScaleData()
    Xtrain = titanicScaler.fitScaleData(Xtrain)
    Xcv = titanicScaler.scaleData(Xcv)
    # pd.DataFrame(scaler.fit_transform(data), columns=data.columns).as_matrix()

    # Support Vector Machine
    import Z_LearnSVC
    svc = Z_LearnSVC.Z_LearnSVC()
    svc.setrbf()
    cvScore, cvPredict, cvC = svc.fitC(Xtrain, yTrain, Xcv, ycv) 
    print(cvScore, cvPredict, cvC)

    # Support Vector Machine
    svc.setsigmoid()
    cvScore, cvPredict, cvC = svc.fitC(Xtrain, yTrain, Xcv, ycv) 
    print(cvScore, cvPredict, cvC)

    # Support Vector Machine
    #svc.setpoly()
    #cvScore, cvPredict, cvC = svc.fitC(Xtrain, yTrain, cvScaled, yCV) 
    #print(cvScore, cvPredict, cvC)

    # Plot Scores
    #svc.setrbf()
    print(len(Xtrain), len(yTrain))
    print(len(Xcv), len(ycv))
    svc.plotScores(Xtrain, yTrain, Xcv, ycv)
        
    # Test Model
    
    # Save Test Predictions
    
    