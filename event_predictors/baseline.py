import sys                                                  #needed to import the helper functions, because that module is outside this directory
sys.path.append(''.join(sys.path[0].split('\\\\')[0:-1]))   #needed to import the helper functions, because that module is outside this directory
from helper_functions import Helper                                     # import our helper functions
import pandas as pd                                         # for data handling
from sklearn.tree import DecisionTreeClassifier             # for decision tree mining


class baseline_events:
    def __init__(self, dataSet) -> None:

        self.data = dataSet
        
        #create decision tree object
        self.dtc = DecisionTreeClassifier(max_depth=1)

        #format data into df for the decision tree
        self.preProcessData()

        #create X with one hot encoding
        self.X_dtc_train = pd.get_dummies(self.trainingSet[['concept:name']],drop_first=True).copy()
        self.X_dtc_train['positionInTrace'] = self.trainingSet['positionInTrace']
        self.y_dtc_train = self.trainingSet[['next_concept:name']].copy()

        #create X with one hot encoding
        self.X_dtc_test = pd.get_dummies(self.testSet[['concept:name']],drop_first=True).copy()
        self.X_dtc_test['positionInTrace'] = self.testSet['positionInTrace']
        self.y_dtc_test = self.testSet[['next_concept:name']].copy()

        #fit the model
        self.dtc.fit(self.X_dtc_train, self.y_dtc_train)


    #helper function for init for readability
    def preProcessData(self):
        self.data['next_concept:name'] = self.data.groupby('@@case_index')['concept:name'].shift(-1).fillna('Finish')
        self.data['traceEndDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('max')
        self.data['traceStartDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('min')
        self.data['minIndexInCase'] = self.data.groupby('@@case_index')['@@index'].transform('min')
        self.data['positionInTrace'] = self.data['@@index'] - self.data['minIndexInCase']
        cutoff = self.data.sort_values('traceEndDate').iloc[int(len(self.data) * 0.82)]['traceEndDate']
        self.trainingSet = self.data.loc[(self.data['traceEndDate'] < cutoff), ['concept:name', 'next_concept:name', 'positionInTrace']]
        self.testSet = self.data.loc[(self.data['traceStartDate'] >= cutoff), ['concept:name', 'next_concept:name', 'positionInTrace']]

    #get statistics of the model
    def info(self):
            print('Goodness of fit:', self.dtc.score(self.X_dtc_train, self.y_dtc_train))
            print('Predictive performance:', self.dtc.score(self.X_dtc_test, self.y_dtc_test))
            print(f'Size of training set size relative to dataset:          {len(self.trainingSet) / len(self.data)}')
            print(f'Size of valdiation set relative to dataset:             {len(self.testSet) / len(self.data)}')
            print(f'Size of validation set relvative to training set:       {len(self.testSet) / len(self.trainingSet)}')

    #predict which events comes after a specific event
    def predict(self, inputEvent, eventTraceIndex, probability=False):
        if 'concept:name_' + inputEvent not in list(self.X_dtc_train):
            raise Exception('concept:name_' + inputEvent, 'not in', list(self.X_dtc_train))
        inputFrame = pd.DataFrame(columns=list(self.X_dtc_train))
        inputArray = ([False] * (len(list(self.X_dtc_train)) - 1))
        inputArray.append(eventTraceIndex)
        inputFrame.loc[0] = inputArray
        inputFrame.loc[0, 'concept:name_' + inputEvent] = True
        if probability:
            return self.dtc.predict(inputFrame)[0], pd.Series(self.dtc.predict_proba(inputFrame)[0]).max()
        return self.dtc.predict(inputFrame)[0]
         