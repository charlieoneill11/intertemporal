import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

def load_data(name="charles_oneill"):
    # load training data
    train = pd.read_csv(f"~/intertemporal/data/{name}_train.csv")
    cols = ["SIR", "LDR", "Delay", "Answer"]
    # load testing data
    test = pd.read_csv(f"~/intertemporal/data/{name}_test.csv")
    return train[cols], test[cols]

class Experiment:
    
    def __init__(self, train, test, model, scale=20):
        self.train = train
        self.test = test
        self.X_train, self.y_train = train.drop(columns=['Answer']), train.Answer.values
        self.X_test, self.y_test = test.drop(columns=['Answer']), test.Answer.values
        self.model = model
        self.scale = scale
        
    def normalise(self):
        trans_train = Normalizer().fit(self.X_train)
        trans_test = Normalizer().fit(self.X_test)
        X_train = trans_train.transform(self.X_train)
        X_test = trans_test.transform(self.X_test)
        return X_train, X_test
    
    def normalise_params(self):
        train, test = self.train.copy(), self.test.copy()
        train.SIR /= self.scale
        train.LDR /= self.scale
        test.SIR /= self.scale
        test.LDR /= self.scale
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

class ExperimentCollection:

    def __init__(self, dataset, test_size=0.25):
        self.data = dataset
        self.X = dataset.drop(columns=['Answer'])
        self.y = dataset['Answer'].values
        self.test_size = test_size

    def cross_val(self, n_iters=10):
        model_accs, param_accs = np.zeros(n_iters), np.zeros(n_iters)
        for i in range(n_iters):
            train, test, y_train, y_test = train_test_split(self.X, self.y, 
                                                                test_size=self.test_size)
            train["Answer"], test["Answer"] = y_train, y_test
            model = xgb.XGBClassifier(verbosity=0)
            exp = Experiment(train, test, model, scale=50)
            model_accs[i], param_accs[i] = exp.run()
        return model_accs, param_accs

class Population:

    def __init__(self, n_iters):
        self.n_iters = n_iters
        cwd = Path.home()
        names_path = cwd.joinpath("intertemporal/data/names.txt")
        file = open(names_path,"r")
        names = file.readlines()
        self.names = [element.strip() for element in names]
    
    def population_run(self):
        model_results = np.zeros((len(self.names), self.n_iters))
        param_results = np.zeros((len(self.names), self.n_iters))
        for i, name in tqdm(enumerate(self.names)):
            train, test = load_data(name)
            df = pd.concat([train, test])
            exp_collect = ExperimentCollection(df)
            model_results[i], param_results[i] = exp_collect.cross_val(n_iters=self.n_iters)
        return model_results, param_results

    def print_results(self):
        model_results, param_results = self.population_run()
        print(np.mean(np.mean(model_results, axis=1)))
        print(np.mean(np.mean(param_results, axis=1)))

if __name__ == "__main__":
    pop = Population(20)
    pop.print_results()
    
"""if __name__ == "__main__":   
    train, test = load_data("max_kirkby")
    df = pd.concat([train, test])
    exp_collect = ExperimentCollection(df)
    m, p = exp_collect.cross_val(20)
    print(np.mean(m), np.mean(p))"""
        
