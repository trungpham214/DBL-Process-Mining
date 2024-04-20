import logging
import sys  # needed to import the helper functions, because that module is outside this directory

sys.path.append(''.join(sys.path[0].split('\\\\')[
                        0:-1]))  # needed to import the helper functions, because that module is outside this directory

from helper_functions import Helper  # import our helper functions
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm  # progress bar for training
import pandas as pd
import numpy as np
from datetime import timedelta

COLUMNLIST = ['positionInTrace', 'concept:name']

class predictor:
    def __init__(self, dataSet, columns) -> None:
        self.data = dataSet
        self.columns = columns
        self.preProcessData()
        self.createDecider()

    # helper function for init for readability
    def preProcessData(self):
        self.data['next_time:timestamp'] = self.data.groupby('@@case_index')['time:timestamp'].shift(-1)
        self.data = self.data[self.data['next_time:timestamp'].notna()].copy()
        self.data['timeDelta'] = self.data['next_time:timestamp'] - self.data['time:timestamp']
        self.data['traceEndDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('max')
        self.data['traceStartDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('min')
        self.data['minIndexInCase'] = self.data.groupby('@@case_index')['@@index'].transform('min')
        self.data['positionInTrace'] = self.data['@@index'] - self.data['minIndexInCase']
        helper = Helper()
        cutoff = helper.findCutoff(self.data)
        cutoff = self.data.sort_values('traceEndDate').iloc[int(len(self.data) * cutoff)]['traceEndDate']
        fields = ['timeDelta'] + self.columns
        self.trainingSet = self.data.loc[(self.data['traceEndDate'] < cutoff), fields]
        self.testSet = self.data.loc[(self.data['traceStartDate'] >= cutoff), fields]

    def createDecider(self):
        self.deciderDict = {}
        for index, row in tqdm(
                self.trainingSet.groupby(self.columns)['timeDelta'].agg(pd.Series.mean).to_frame().iterrows(),
                leave=False, desc='Training model'):
            value = row['timeDelta']
            if type(value) == pd.core.arrays.string_.StringArray:
                value = value[0]
            if not type(row.name) == tuple:
                row.name = row.name,
            conditions = []
            for i in range(0, len(self.columns)):
                conditions.append(self.trainingSet[self.columns[i]] == row.name[i])

            self.deciderDict[row.name] = (value, len(self.trainingSet.loc[
                                                         (self.trainingSet['timeDelta'] == value)
                                                         & (np.logical_and.reduce(conditions))]) /
                                          len(np.logical_and.reduce(conditions)))

    # get statistics of the model
    def info(self):

        print(f'Goodness of fit:                                        {self.getGoodnessOfFit()}')
        print(f'Predictive performance:                                 {self.getPredictivePerformance()}')
        print(f'Size of training set size relative to dataset:          {len(self.trainingSet) / len(self.data)}')
        print(f'Size of test set relative to dataset:                   {len(self.testSet) / len(self.data)}')
        print(f'Size of test set relvative to training set:             {len(self.testSet) / len(self.trainingSet)}')
        #print(f'MAE and RMSE and MAPE:                                  {self.evaluate_predictor(self.trainingSet)}')

    def getGoodnessOfFit(self):
        correct = 0
        for i in tqdm(self.deciderDict.keys(), leave=False, desc='Getting goodness of fit'):
            conditions = []
            for j in range(0, len(self.columns)):
                conditions.append(self.trainingSet[self.columns[j]] == i[j])
            value = self.deciderDict.get(i)[0]
            correct += len(self.trainingSet.loc[
                               (self.trainingSet['timeDelta'] - timedelta(15) <= value)
                                & (self.trainingSet['timeDelta'] + timedelta(15) >= value)
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
                                (self.testSet['timeDelta'] - timedelta(15) <= value)
                                & (self.testSet['timeDelta'] + timedelta(15) >= value)
                                & (np.logical_and.reduce(conditions))])

        return correct / len(self.testSet)

    def printProbabilities(self):
        for i in self.deciderDict.keys():
            print(i, '\n\t', self.deciderDict.get(i))

    # predict which time to next event
    def predict(self, key: tuple):
        prediction = self.deciderDict.get(key, (None,))
        return prediction[0]

    def predictAll(self):
        return self.data.apply(lambda row: self.extract_values(row), axis=1)

    def extract_values(self, row):
        key = tuple(row[col] for col in self.columns)
        value = self.deciderDict.get(key, (None,))
        return value[0]

    def evaluate_predictor(self, test_data):
        all_predicted_events_df = self.predictAll()
        predicted_events_df = test_data.apply(
            lambda row: self.extract_values(row), axis=1)
        actual_events = test_data['concept:name']

        actual_events['next_time:timestamp'] = actual_events['next_time:timestamp'].dt.total_seconds()

        # Calculate MAE and RMSE
        mae = mean_absolute_error(actual_events['timeDelta'], predicted_events_df['timeDelta'])
        rmse = mean_squared_error(actual_events['timeDelta'], predicted_events_df['timeDelta'],
                                  squared=False)
        mask = actual_events['timeDelta'] != 0
        mape_values = np.abs(
            (actual_events.loc[mask, 'timeDelta'] - predicted_events_df.loc[mask, 'timeDelta']) /
            actual_events.loc[mask, 'timeDelta']) * 100
        mape = np.mean(mape_values)
        return mae, rmse, mape
