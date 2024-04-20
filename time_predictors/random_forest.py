import sys

import numpy

sys.path.append(''.join(sys.path[0].split('\\\\')[
                        0:-1]))  # needed to import the helper functions, because that module is outside this directory
from helper_functions import Helper  # import our helper functions
import pandas as pd  # data processing
from sklearn.ensemble import RandomForestRegressor  # randomforest
from sklearn.metrics import r2_score
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from tqdm import tqdm
# from event_predictors.random_forest import predictor as ev_predictor

COLUMNLIST = ['concept:name', 'parallel_cases']
RANDOMFORESTPARAMS = {
    'verbose': 1,
    'n_jobs': -1,
    'n_estimators' : 750
}
PARAMDIST = {"max_depth": [3, None],
             "min_samples_split": sp_randint(2, 11),
             "min_samples_leaf": sp_randint(1, 11),
             "n_estimators": sp_randint(100, 1000)}
RSCVPARAMS = {
    'verbose': 3,
    'n_jobs': -1,
    'estimator': RandomForestRegressor(**RANDOMFORESTPARAMS),
    'n_iter': 1,
    'cv': 1,
    'random_state': 1,
    'param_distributions': PARAMDIST
}


class MockEventPredictor:
    def __init__(self, data):
        self.data = data

    def predict(self, prefix):
        return self.data[(self.data['@@case_index'] < 100)][0:-1]

class predictor:
    def __init__(self, dataSet, columns) -> None:
        self.data = dataSet
        self.columns = columns
        self.preProcessData()
        self.genRandomForest()

    # helper function for init for readability
    def preProcessData(self) -> None:
        self.data['next_concept:name'] = self.data.groupby('@@case_index')['concept:name'].shift(-1).fillna('Finish')
        self.data['next_time:timestamp'] = self.data.groupby('@@case_index')['time:timestamp'].shift(-1)
        self.data['time_to_next'] = self.data['next_time:timestamp'] - self.data['time:timestamp']

        # We add a column with the number of cases in parallel by an organizer
        self.data['org:resource'] = self.data.groupby('case:concept:name')['org:resource'].ffill()
        self.data = self.data.sort_values(by=['time:timestamp', 'time_to_next'])
        for org, org_group in self.data.groupby('org:resource'):
            org_group = org_group.sort_values(by=['time:timestamp'])
            org_group['parallel_cases'] = (org_group['concept:name'] == 'Create Fine').cumsum() - (
                    org_group['next_concept:name'] == 'Finish').cumsum()
            self.data.loc[org_group.index, 'parallel_cases'] = org_group['parallel_cases']
        self.data = self.data.sort_values(by=['@@index'])

        self.data = self.data[self.data['next_time:timestamp'].notna()].copy()
        self.data['traceEndDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('max')
        self.data['traceStartDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('min')
        self.data['minIndexInCase'] = self.data.groupby('@@case_index')['@@index'].transform('min')
        self.data['positionInTrace'] = self.data['@@index'] - self.data['minIndexInCase']
        self.data['month'] = self.data['time:timestamp'].dt.month

        helper = Helper()
        cutoff = helper.findCutoff(self.data)

        dataDummies = pd.get_dummies(self.data[self.columns + ['traceStartDate', 'traceEndDate']])

        self.trainingSetDummies = dataDummies.loc[(dataDummies['traceEndDate'] < cutoff)].drop(
            columns=['traceEndDate', 'traceStartDate'])
        self.testSetDummies = dataDummies.loc[(dataDummies['traceStartDate'] >= cutoff)].drop(
            columns=['traceEndDate', 'traceStartDate'])

        self.trainingSetResults = self.data.loc[(self.data['traceEndDate'] < cutoff), ['time_to_next']]
        self.trainingSetResults = self.trainingSetResults['time_to_next'].dt.total_seconds()
        self.testSetResults = self.data.loc[(self.data['traceStartDate'] >= cutoff), ['time_to_next']]
        self.testSetResults = self.testSetResults['time_to_next'].dt.total_seconds()

    def genRandomForest(self) -> None:
        self.rf = RandomForestRegressor(**RANDOMFORESTPARAMS, random_state=1)
        self.trainingSetDummies = self.trainingSetDummies.reindex(sorted(self.trainingSetDummies.columns), axis=1)
        self.rf.fit(self.trainingSetDummies, self.trainingSetResults.squeeze())

    # get statistics of the model
    def info(self) -> None:
        print(f'Random forest goodness of fit:                          {self.getGoodnessOfFitRF()}')
        print(f'Random forest Predictive performance:                   {self.getPredictivePerformanceRF()}')
        print(
            f'Size of training set size relative to dataset:          {len(self.trainingSetResults) / len(self.data)}')
        print(f'Size of test set relative to dataset:                   {len(self.testSetResults) / len(self.data)}')
        print(
            f'Size of test set relvative to training set:             {len(self.testSetResults) / len(self.trainingSetResults)}')

    def getGoodnessOfFitRF(self) -> float:
        predictions = self.rf.predict(self.trainingSetDummies)
        return r2_score(self.trainingSetResults, predictions)

    def getPredictivePerformanceRF(self) -> float:
        predictions = self.rf.predict(self.testSetDummies)
        return r2_score(self.testSetResults, predictions)

    def predictAll(self):
        times_to_next = self.rf.predict(pd.get_dummies(self.data[self.columns]))
        current_timestamps = self.data['time:timestamp'].to_list()
        timedeltas = [timedelta(seconds=time_to_next) for time_to_next in times_to_next]
        predictions = [current_timestamps[i] + timedeltas[i] for i in range(len(current_timestamps))]
        return predictions

    def predict_suffix(self, ev_suffixes: pd.DataFrame):
        # Assumption: Each row of ev_suffixes contains a prefix completed with event suffix.
        # We iterate over each row (trace) to predict the times.
        # We also assume the values of any relevant columns for the trace are provided
        # in columns of the same name.

        # The array containing suffix predictions for each trace
        predictions = []

        for index, row in tqdm(ev_suffixes.iterrows(), desc="Time predictions", leave=False):
            
            # Deconstructing the event and parallel cases
            ev_suffix = row['suffix']
            if len(ev_suffix) == 0:
                predictions.append([])
                continue
            parallel_cases = [row['parallel_cases'] for i in range(len(ev_suffix))]
            last_timestamp = row['time:timestamp']

            # Building a dataframe with events and parallel cases
            df = pd.DataFrame(data={'concept:name': ev_suffix, 'parallel_cases': parallel_cases})

            # Getting dummies for trace and making encoding consistent
            dummies = pd.get_dummies(df[self.columns])
            for column in self.testSetDummies.columns.difference(dummies.columns):
                dummies[column] = False

            dummies = dummies.reindex(sorted(dummies.columns), axis=1)

            # Make predictions
            times_to_next = self.rf.predict(dummies)
            timedeltas = [timedelta(seconds=time_to_next) for time_to_next in times_to_next]
            prediction = [last_timestamp + sum(timedeltas[:i + 1], timedelta()) for i in range(len(timedeltas))]
            predictions.append(prediction)

        suffixes = ev_suffixes
        suffixes['predicted_next_time:timestamp'] = predictions
        return suffixes
    

    def predict_suffix1(self, ev_suffixes: pd.DataFrame):
        # Assumption: Each row of ev_suffixes contains a prefix completed with event suffix.
        # We iterate over each row (trace) to predict the times.
        # We also assume the values of any relevant columns for the trace are provided
        # in columns of the same name.

        # The array containing suffix predictions for each trace
        ev_suffixes = ev_suffixes[['parallel_cases', 'suffix', 'time:timestamp', '@@case_index']].explode('suffix')
        ev_suffixes = ev_suffixes.rename(columns={'suffix' : 'concept:name'})
        ev_suffixes = ev_suffixes[ev_suffixes['concept:name'].notna()]
        dummies = pd.get_dummies(ev_suffixes[self.columns])
        for column in self.testSetDummies.columns.difference(dummies.columns):
            dummies[column] = False

        dummies = dummies.reindex(sorted(dummies.columns), axis=1)

        ev_suffixes['predicted'] = self.rf.predict(dummies) * timedelta(seconds=1)

        ev_suffixes['predicted'] = ev_suffixes.groupby('@@case_index')['predicted'].cumsum()
        ev_suffixes['time:timestamp'] = ev_suffixes['time:timestamp'] + ev_suffixes['predicted']
        return ev_suffixes[['concept:name', 'time:timestamp']]