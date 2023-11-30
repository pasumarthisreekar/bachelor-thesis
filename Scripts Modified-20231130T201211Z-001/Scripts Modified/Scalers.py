import numpy as np

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DataTypeError(Error):
    """Raised when the input data type is invalid"""
    pass

class ScalerNotFitError(Error):
    """Raised when the scaler object has not been fit to data"""
    pass

class GlobalMinMaxScaler:
    
    def __init__(self, feature_range=(0., 1.)):
        self._min, self._max = feature_range
        self._dmin, self._dmax = None, None
        
        
    def __checkdatatype(self, X):
        try:
            assert type(X) == np.ndarray
        except AssertionError:
            print('Data should be a numpy array')            
            raise DataTypeError
        
        
    def __checkfit(self):
        if self._dmin is None or self._dmax is None:
            print("GlobalMinMaxScaler object has not been fit to data")
            raise ScalerNotFitError
        
        
    def fit(self, X):
        self._dmin = X.min()
        self._dmax = X.max()
        
        
    def transform(self, X):
        
        self.__checkfit()
        self.__checkdatatype(X)
        
        X_std = (X - self._dmin) / (self._dmax - self._dmin)
        X_scaled = X_std * (self._max - self._min) + self._min
        
        return X_scaled
        
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
    def inverse_transform(self, X):
        pass