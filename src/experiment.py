import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

from parameter_fit import *

import warnings
warnings.filterwarnings("ignore")

class Experiment:
    
    def __init__(self, train, test, model):
        self.train = train
        self.test = test
        self.X_train, self.y_train = train.drop(columns=['Answer']), train.Answer.values
        self.X_test, self.y_test = test.drop(columns=['Answer']), test.Answer.values
        self.model = model
        
    def normalise(self):
        trans_train = Normalizer().fit(self.X_train)
        trans_test = Normalizer().fit(self.X_test)
        X_train = trans_train.transform(self.X_train)
        X_test = trans_test.transform(self.X_test)
        return X_train, X_test
    
    def normalise_params(self):
        train, test = self.train.copy(), self.test.copy()
        train.SIR /= 10
        train.LDR /= 10
        test.SIR /= 10
        test.LDR /= 10
        return train, test
    
    def run(self):
        # run Model X
        X_train, X_test = self.normalise()
        self.model.fit(X_train, self.y_train)
        model_preds = self.model.predict(X_test)
        model_accuracy = accuracy_score(model_preds, self.y_test)
        # run ParameterFit
        train, test = self.normalise_params()
        param = ParameterFit(train, test)
        param_preds = param.fit()
        param_accuracy = accuracy_score(param_preds, self.y_test)
        return model_accuracy, param_accuracy