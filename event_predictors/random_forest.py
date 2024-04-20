import sys

import numpy                                                  #needed to import the helper functions, because that module is outside this directory
sys.path.append(''.join(sys.path[0].split('\\\\')[0:-1]))   #needed to import the helper functions, because that module is outside this directory
from helper_functions import Helper                         #import our helper functions
import pandas as pd                                         #data processing
from sklearn.ensemble import RandomForestClassifier         #randomforest
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import ast



COLUMNLIST = ['article', 'amount', 'prefix', 'parallel_cases'] # TODO : Change this
RANDOMFORESTPARAMS = {
     'verbose' : 1,
     'n_jobs'  : -1,
     'n_estimators' : 750,
     'max_depth' : 7
}
PARAMDIST = {}
RSCVPARAMS = {
     'verbose'      : 3,
     'n_jobs'       : 4,
     'estimator'    : RandomForestClassifier(**RANDOMFORESTPARAMS),
     'n_iter'       : 1,
     'cv'           : 5,
     'random_state' : 1,
     'param_distributions' : PARAMDIST
}

class predictor:

    def __init__(self, dataSet, columns) -> None:
        self.columns = columns
        self.data = self.preProcessData(dataSet, True)
        self.genRandomForest()

    # helper function for init for readability
    def preProcessData(self, data, forModel=False) -> None:
        data['next_concept:name'] = data.groupby('@@case_index')['concept:name'].shift(-1).fillna('Finish')
        data['traceEndDate'] = data.groupby('@@case_index')['time:timestamp'].transform('max')
        data['traceStartDate'] = data.groupby('@@case_index')['time:timestamp'].transform('min')
        data['minIndexInCase'] = data.groupby('@@case_index')['@@index'].transform('min')
        data['org:resource'] = data['org:resource'].astype('Int64')
        data['org:resource'] = data.groupby('@@case_index')['org:resource'].transform('min')
        data['positionInTrace'] = data['@@index'] - data['minIndexInCase']
        data['month'] = data['time:timestamp'].dt.month
        data['prefix'] = [(str(y['concept:name'].tolist()[:z+1]), str(y['concept:name'].tolist()[z+1:])) for x, y in tqdm(data.groupby('@@case_index'), leave=False, desc="adding prefix/suffix") for z in range(len(y))] # Note: prefix is inclusive, i.e. the current event is included in the trace.
        data[['prefix', 'suffix']] = pd.DataFrame(data['prefix'].tolist(), index=data.index)
        for org, org_group in data.groupby('org:resource'):
            org_group = org_group.sort_values(by=['time:timestamp'])
            org_group['parallel_cases'] = (org_group['concept:name'] == 'Create Fine').cumsum() - (
                    org_group['next_concept:name'] == 'Finish').cumsum()
            data.loc[org_group.index, 'parallel_cases'] = org_group['parallel_cases']

        data.update(data.groupby("@@case_index").ffill())

        if forModel:
            helper = Helper()
            cutoff = helper.findCutoff(data)

            self.prefixEncoder = defaultdict(LabelEncoder)
            data['prefix'] = data[['prefix']].apply(lambda x: self.prefixEncoder[x.name].fit_transform(x))

            dataDummies = pd.get_dummies(data[self.columns + ['traceStartDate', 'traceEndDate']])
            dataDummies = dataDummies.reindex(sorted(dataDummies.columns), axis=1)
            
            self.suffixEncoder = defaultdict(LabelEncoder)
            
            suffixEncoded = data[['traceStartDate', 'traceEndDate', 'suffix']].copy()
            suffixEncoded['suffix'] = suffixEncoded[['suffix']].apply(lambda x: self.suffixEncoder[x.name].fit_transform(x))

            self.trainingSetEncoded = dataDummies.loc[(dataDummies['traceEndDate'] < cutoff)].drop(columns=['traceEndDate', 'traceStartDate'])
            self.testSetEncoded = dataDummies.loc[(dataDummies['traceStartDate'] >= cutoff)].drop(columns=['traceEndDate', 'traceStartDate'])

            self.trainingSetResults = suffixEncoded.loc[(suffixEncoded['traceEndDate'] < cutoff), ['suffix']]
            self.testSetResults = suffixEncoded.loc[(suffixEncoded['traceStartDate'] >= cutoff), ['suffix']]

        
        return data

    def genRandomForest(self) -> None:
        #rscv = RandomizedSearchCV(**RSCVPARAMS)
        #rscv.fit(self.trainingSetEncoded, self.trainingSetResults['suffix'])
        #self.rscv = rscv.best_estimator_
        self.rf = RandomForestClassifier(**RANDOMFORESTPARAMS, random_state=1)
        self.rf.fit(self.trainingSetEncoded, self.trainingSetResults['suffix'])
        
        del self.testSetEncoded
        del self.testSetResults
        del self.trainingSetResults

    #get statistics of the model
    def info(self) -> None:
            #print(f'Random forest goodness of fit:                          {self.getGoodnessOfFitRF()}')
            #print(f'Random forest Predictive performance:                   {self.getPredictivePerformanceRF()}')
            #print(f'RSCV goodness of fit:                                   {self.getGoodnessOfFitRSCV()}')
            #print(f'RSCV Predictive performance:                            {self.getPredictivePerformanceRSCV()}')
            print(f'Size of training set size relative to dataset:          {len(self.trainingSetResults) / len(self.data)}')
            print(f'Size of test set relative to dataset:                   {len(self.testSetResults) / len(self.data)}')
            print(f'Size of test set relvative to training set:             {len(self.testSetResults) / len(self.trainingSetResults)}')


    def getGoodnessOfFitRF(self) -> float:
        predictions = self.rf.predict(self.trainingSetEncoded)
        return accuracy_score(self.trainingSetResults, predictions)
    
    def getPredictivePerformanceRF(self) -> float:
        predictions = self.rf.predict(self.testSetEncoded)
        return accuracy_score(self.testSetResults, predictions)
    
    def getGoodnessOfFitRSCV(self) -> float:
        predictions = self.rscv.predict(self.trainingSetEncoded)
        return accuracy_score(self.trainingSetResults, predictions)
    
    def getPredictivePerformanceRSCV(self) -> float:
        predictions = self.rscv.predict(self.testSetEncoded)
        return accuracy_score(self.testSetResults, predictions)
        
    def predictAll(self) -> numpy.ndarray:
        data = self.data[self.columns].copy()
        data = pd.get_dummies(data)
        data = data.reindex(sorted(data.columns), axis=1)
        dummies = pd.get_dummies(data)
        
        results = []
        i = 0
        while i < len(dummies):
            j = i + 50000
            if j > len(dummies):
                j = len(dummies)
            results += self.rf.predict(dummies.iloc[i:j]).tolist()
            i = i + 50000
        predictions = pd.DataFrame(results, columns=['suffix']).apply(lambda x: self.suffixEncoder[x.name].inverse_transform(x))['suffix']
        predictions = predictions.apply(lambda x: ast.literal_eval(x))
        return predictions
    
    def predict(self, trace):
        
        trace = self.preProcessData(trace)

        
        trace = trace[(trace['suffix'] == '[]')].copy()
        traceCopy = trace.copy()

        trace['prefix'] = trace[['prefix']].apply(lambda x: self.prefixEncoder[x.name].transform(x))
        trace = pd.get_dummies(trace[self.columns])
        for i in set(self.trainingSetEncoded) - set(trace):
            trace[i] = False
        trace = trace.reindex(sorted(trace.columns), axis=1)
        predicted = []
        i = 0
        while i < len(trace):
            j = i + 50000
            if j > len(trace):
                j = len(trace)
            predicted += self.rf.predict(trace.iloc[i:j]).tolist()
            i = i + 50000
        traceCopy['suffix'] = pd.DataFrame(predicted, columns=['suffix']).apply(lambda x: self.suffixEncoder[x.name].inverse_transform(x))['suffix'].apply(lambda x: ast.literal_eval(x)).values
        return traceCopy
