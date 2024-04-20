import pm4py
import pandas as pd

class Helper:
    def __init__(self) -> None:
        self.data = {}

    def loadDataPKL(self) -> pd.DataFrame:
        return pd.read_pickle('dataset.pkl')
    
    def loadDataXES(self, path:str) -> pd.DataFrame:
        if path not in self.data:
            event_log = pm4py.format_dataframe(pm4py.read_xes(path))
            df = pd.DataFrame(event_log)
            self.data[path] = df
            return df.copy()
        else:
            return self.data.get(path).copy()


    # Split data into training and test.
    # Pre: data contains traceEndDate and traceStartDate columns
    def split_data(self, data):
        cutoff = data.sort_values('traceEndDate').iloc[int(len(data) * 0.82)]['traceEndDate']

        df_train = data.loc[data['traceEndDate'] < cutoff]
        df_test = data.loc[data['traceStartDate'] >= cutoff]
        return df_train, df_test
    

    #find the cutoff time for a dataset given a margin, provided it has traceEndDate and traceStartDate columns
    #will attempt to get a 9:1 ratio. The margin is s.t. size(test)/size(training) = (1/9) +- margin, so the margin dictates how many percent the actual sizes may be off from the 9:1 ratio
    def findCutoff(self, dataSet, margin=0.1):
        self.dataSet = dataSet
        self.margin = margin
        self.upperbound = 1
        self.lowerbound = 0
        self.val = 0.5
        self._findCutOff()
        return dataSet.sort_values('traceEndDate').iloc[int(len(dataSet) * self.val)]['traceEndDate']
        
    def _findCutOff(self):
        cutoff = self.dataSet.sort_values('traceEndDate').iloc[int(len(self.dataSet) * self.val)]['traceEndDate']
        df_train = self.dataSet.loc[self.dataSet['traceEndDate'] < cutoff]
        df_test = self.dataSet.loc[self.dataSet['traceStartDate'] >= cutoff]
      
        if ((len(df_test) / len(df_train) >= ((1/9) - self.margin/100))
            and (len(df_test) / len(df_train) <= ((1/9) + self.margin/100))):
            pass
        else:
            if len(df_test) / len(df_train) >= ((1/9) - self.margin/100):
                self.lowerbound = self.val
                self.val = (self.lowerbound + self.upperbound)/2
                self._findCutOff()
            else:
                self.upperbound = self.val
                self.val = (self.lowerbound + self.upperbound)/2
                self._findCutOff()