'''
Created on Apr 7, 2017

@author: dave
'''

class ScaleData(object):
    '''
    classdocs
    '''
    from sklearn.preprocessing import MinMaxScaler
    
    def __init__(self, scaler=MinMaxScaler(copy=False)):
        '''
        Constructor
        '''
        self.scaler = scaler
    
    def fitData(self, data):
        self.scaler.fit(data)
        
    def fitScaleData(self, data):
        return self.scaler.fit_transform(data)

    def scaleData(self, data):
        return self.scaler.transform(data)
        
    def setScaler(self, scaler):
        self.scaler = scaler
        
    def getScaler(self):
        return self.scaler
    
    def saveScaler(self, filename):
        import pickle
        with open('filename', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.scaler], f)

    def loadScaler(self, filename):
        import pickle
        with open('filename') as f:  # Python 3: open(..., 'wb')
            self.scaler = pickle.load(f)
        
        
        
