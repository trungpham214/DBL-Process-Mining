import sys
import numpy as np
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

COLUMNLIST = ['concept:name', 'parallel_cases', 'totalPaymentAmount', 'expense'] # TODO : Change this
RANDOMFORESTPARAMS = {
    'verbose': 1,
    'n_jobs': -1
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
        cutoff = self.data.sort_values('traceEndDate').iloc[int(len(self.data) * cutoff)]['traceEndDate']
        
        dataDummies = pd.get_dummies(self.data[self.columns + ['traceStartDate', 'traceEndDate']])
        
        self.trainingSetDummies = dataDummies.loc[(dataDummies['traceEndDate'] < cutoff)].drop(columns=['traceEndDate', 'traceStartDate'])
        self.testSetDummies = dataDummies.loc[(dataDummies['traceStartDate'] >= cutoff)].drop(columns=['traceEndDate', 'traceStartDate'])
        
        self.trainingSetResults = self.data.loc[(self.data['traceEndDate'] < cutoff), ['time_to_next']]
        self.trainingSetResults = self.trainingSetResults['time_to_next'].dt.total_seconds()
        self.testSetResults = self.data.loc[(self.data['traceStartDate'] >= cutoff), ['time_to_next']]
        self.testSetResults = self.testSetResults['time_to_next'].dt.total_seconds()

    def genRandomForest(self) -> None:
        self.rf = RandomForestRegressor(**RANDOMFORESTPARAMS, random_state=1)
        self.rf.fit(self.trainingSetDummies, self.trainingSetResults.squeeze())

    # get statistics of the model
    def info(self) -> None:
        print(f'Random forest goodness of fit:                          {self.getGoodnessOfFitRF()}')
        print(f'Random forest Predictive performance:                   {self.getPredictivePerformanceRF()}')
        print(f'MAE on test set:                                        {self.meanAbsoluteErrorTest()}')
        print(f'MAE on train set:                                       {self.meanAbsoluteErrorTrain()}')
        print(f'MSE on train set:                                       {self.meanSquaredErrorTrain()}')
        print(f'MAE on test set:                                        {self.meanSquaredErrorTest()}')
        print(f'RMSE on train set:                                      {self.rootMeanSquaredErrorTrain()}')
        print(f'RMSE on test set:                                       {self.rootMeanSquaredErrorTest()}')
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

    def meanAbsoluteErrorTrain(self) -> float:
        predictions = self.rf.predict(self.trainingSetDummies)
        return mean_absolute_error(self.trainingSetResults, predictions)

    def meanAbsoluteErrorTest(self) -> float:
        predictions = self.rf.predict(self.testSetDummies)
        return mean_absolute_error(self.testSetResults, predictions)

    def meanSquaredErrorTrain(self) -> float:
        predictions = self.rf.predict(self.trainingSetDummies)
        return mean_squared_error(self.trainingSetResults, predictions)

    def meanSquaredErrorTest(self) -> float:
        predictions = self.rf.predict(self.testSetDummies)
        return mean_squared_error(self.testSetResults, predictions)

    def rootMeanSquaredErrorTrain(self) -> float:
        predictions = self.rf.predict(self.trainingSetDummies)
        mse_train = mean_squared_error(self.trainingSetResults, predictions)
        return  np.sqrt(mse_train)

    def rootMeanSquaredErrorTest(self) -> float:
        predictions = self.rf.predict(self.testSetDummies)
        mse_test = mean_squared_error(self.testSetResults, predictions)
        return  np.sqrt(mse_test)

    def predictAll(self):
        times_to_next = self.rf.predict(pd.get_dummies(self.data[self.columns]))
        current_timestamps = self.data['time:timestamp'].to_list()
        timedeltas = [timedelta(seconds=time_to_next) for time_to_next in times_to_next]
        predictions = [current_timestamps[i] + timedeltas[i] for i in range(len(current_timestamps))]
        return predictions


# helper = Helper()
# dataset = helper.loadData('Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz')
dataset = pd.read_pickle('dataset.pkl')
p = predictor(dataset, COLUMNLIST)
p.info()
# uncomment the following line to show all prediction at your own peril (411100 predictions)!
# print(p.predictAll())
