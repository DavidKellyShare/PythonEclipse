'''
Created on Apr 6, 2017

@author: dave
'''

def CleanData(data):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Clean Cabin
    CabinId={'A':1,'B':2,'C':3,'D':4,'E':5, '2':0}
    data['CabinType'] = data['Cabin'].str.extract('^([ABCDE]).*', expand=False).apply(lambda x: CabinId.get(x,6))
    data['CabinNum'] = data['Cabin'].str.extract('([0123456789]+)', expand=False).fillna(0)
    del data['Cabin']
    
    # Clean Embarked
    Embark = {'S':1,'C':2,'Q':3}
    data['EmbarkId'] = data['Embarked'].apply(lambda x: Embark.get(x,0))
    #data['Embarked'].unique()
    del data['Embarked']
    
    # Clean Fare
    #data['Fare'].unique()
    
    # Clean Ticket
    del data['Ticket']
    
    # Clean Age
    #data['Age'].unique()
    #data[data['Age'].isnull()]
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    
    # Clean Sex
    #data['Sex'].unique()
    Sex = {'male':1,'female':2}
    data['SexId'] = data['Sex'].apply(lambda x: Sex.get(x,0))
    del data['Sex']
    
    # Clean Name
    #data['Name'].unique()
    del data['Name']
    
    #data['Pclass'].unique()
    
    
    # Scale Features
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns).as_matrix()
