import sys                                                  #needed to import the helper functions, because that module is outside this directory
sys.path.append(''.join(sys.path[0].split('\\\\')[0:-1]))   #needed to import the helper functions, because that module is outside this directory
from tqdm import tqdm                                       #progress bar for training
import pandas as pd
import numpy as np
from helper_functions import Helper

COLUMNLIST = ['positionInTrace', 'concept:name']
default_path = "Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"

class predictor:
    def __init__(self, dataSet, columns) -> None:
        self.data = dataSet
        self.columns = columns
        self.preProcessData()
        self.createDecider()


    #helper function for init for readability
    def preProcessData(self):
        self.data['next_concept:name'] = self.data.groupby('@@case_index')['concept:name'].shift(-1).fillna('Finish')
        self.data['traceEndDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('max')
        self.data['traceStartDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('min')
        self.data['minIndexInCase'] = self.data.groupby('@@case_index')['@@index'].transform('min')
        self.data['positionInTrace'] = self.data['@@index'] - self.data['minIndexInCase']
        helper = Helper()
        cutoff = helper.findCutoff(self.data)
        cutoff = self.data.sort_values('traceEndDate').iloc[int(len(self.data) * cutoff)]['traceEndDate']
        fields = ['next_concept:name'] + self.columns
        self.trainingSet = self.data.loc[(self.data['traceEndDate'] < cutoff), fields]
        self.testSet = self.data.loc[(self.data['traceStartDate'] >= cutoff), fields]


    def createDecider(self):
        self.deciderDict = {}
        for index, row in tqdm(self.trainingSet.groupby(self.columns)['next_concept:name'].agg(pd.Series.mode).to_frame().iterrows(), leave=False, desc='Training model'):
            value = row['next_concept:name']
            if type(value) == pd.core.arrays.string_.StringArray:
                value = value[0]
            if not type(row.name) == tuple:
                row.name = row.name,
            conditions = []
            for i in range(0, len(self.columns)):
                conditions.append(self.trainingSet[self.columns[i]] == row.name[i])
            
            self.deciderDict[row.name] = (value, len(self.trainingSet.loc[
                                (self.trainingSet['next_concept:name'] == value)
                                & (np.logical_and.reduce(conditions))]) / 
                                len(np.logical_and.reduce(conditions)))
             
    #get statistics of the model
    def info(self):
            print(f'Goodness of fit:                                        {self.getGoodnessOfFit()}')
            print(f'Predictive performance:                                 {self.getPredictivePerformance()}')
            print(f'Size of training set size relative to dataset:          {len(self.trainingSet) / len(self.data)}')
            print(f'Size of test set relative to dataset:                   {len(self.testSet) / len(self.data)}')
            print(f'Size of test set relvative to training set:             {len(self.testSet) / len(self.trainingSet)}')


    def getGoodnessOfFit(self):
        correct = 0
        for i in tqdm(self.deciderDict.keys(), leave=False, desc='Getting goodness of fit'):
            conditions = []
            for j in range(0, len(self.columns)):
                conditions.append(self.trainingSet[self.columns[j]] == i[j])
            value = self.deciderDict.get(i)[0]
            correct += len(self.trainingSet.loc[
                                (self.trainingSet['next_concept:name'] == value)
                                & (np.logical_and.reduce(conditions))])
            
        return correct / len(self.trainingSet)
    
    def getPredictivePerformance(self):
        correct = 0
        for i in tqdm(self.deciderDict.keys(), leave=False, desc='Getting predictive performance'):
            conditions = []
            for j in range(0, len(self.columns)):
                conditions.append(self.testSet[self.columns[j]] == i[j])
            value = self.deciderDict.get(i)[0]
            correct += len(self.testSet.loc[
                                (self.testSet['next_concept:name'] == value)
                                & (np.logical_and.reduce(conditions))])
        return correct / len(self.testSet)
    
    def printProbabilities(self):
        for i in self.deciderDict.keys():
            print(i, '\n\t', self.deciderDict.get(i))
    

    #predict which events comes after a specific event
    def predict(self, key:tuple):
        prediction = self.deciderDict.get(key, (None,))
        return prediction[0]

    def predictAll(self):
        return self.data.apply(lambda row: self.extract_values(row), axis=1)

    def extract_values(self, row):
        key = tuple(row[col] for col in self.columns)
        value = self.deciderDict.get(key, (None,))
        return value[0]
