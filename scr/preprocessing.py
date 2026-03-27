import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    return X, y