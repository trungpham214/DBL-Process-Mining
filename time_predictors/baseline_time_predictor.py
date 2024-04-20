# Baseline prediction for the time between events. We determine the average time between
# event i and i + 1 over all traces and use it as our prediction.
import pandas as pd
from pm4py import convert_to_event_log
from datetime import timedelta
from helper_functions import Helper

class BaselineTimePredictor:
    
    #Construct a baseline time predictor with a path to the event log file
    def __init__(self, data):
        self.data = data.copy()

        df_train, df_test = self._preProcessData()

        self.event_log_train = convert_to_event_log(df_train)
        self.event_log_test = convert_to_event_log(df_test)

        self._create_predictions_arr(self.event_log_train)

    def predict(self, this_event_timestamp, this_event_index):
        if this_event_index < len(self.diff_predictions):
            return this_event_timestamp + timedelta(seconds = self.diff_predictions[this_event_index])
        else:
            avg = sum(self.diff_predictions)/len(self.diff_predictions)
            return this_event_timestamp + timedelta(seconds = avg) # if a trace in the test data is longer than any trace in the training data, return avg of avgs. Maybe handle this better?

    # Return a series object containing all predictions for the data
    def predict_all(self):
        predictions = []
        for index, row in self.data.iterrows():
            this_event_timestamp = row['time:timestamp']
            this_event_index = row['positionInTrace']
            predictions.append(self.predict(this_event_timestamp, this_event_index))
        return pd.Series(predictions)
    def accuracy(self): # Wtime_predictorsORK IN PROGRESS
        predictions = [self.predict(event['time:timestamp'], index) for trace in self.event_log_test for index, event in enumerate(trace)]
        actual_timestamps = [event['time:timestamp'] for trace in self.event_log_test for event in trace]
        diffs = [predictions[i] - actual_timestamps[i] for i in range(len(predictions))]
        within_threshold_count = len([diff for diff in diffs if diff < timedelta(hours = 1000)])
        return within_threshold_count/len(diffs) #return the average difference between actual and predicted
    
    #Helper Methods for this class

    def _preProcessData(self):
        self.data['next_time:timestamp'] = self.data.groupby('@@case_index')['time:timestamp'].shift(-1)

        # Store the start and end dates of the trace in a new column
        self.data['traceEndDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('max')
        self.data['traceStartDate'] = self.data.groupby('@@case_index')['time:timestamp'].transform('min')

        self.data['minIndexInCase'] = self.data.groupby('@@case_index')['@@index'].transform('min')
        self.data['positionInTrace'] = self.data['@@index'] - self.data['minIndexInCase']

        return Helper().split_data(self.data)

    # Create predictions array from event log
    def _create_predictions_arr(self, log):
        # The average time difference between the i'th and i+1'th event is stored at index i of the following array.

        avg_time_differences = []

        # We calculate average using total time differences and counts

        total_time_diffs = []
        counts = []

        for trace in log:
            for i in range(len(trace) - 1):
                this_event = trace[i]
                next_event = trace[i + 1]
                time_diff = (next_event['time:timestamp'] - this_event['time:timestamp']).total_seconds()
                if i < len(total_time_diffs): # at every stage counts and total_time_diffs should have the same length so we only check one
                    total_time_diffs[i] += time_diff
                    counts[i] += 1
                else:
                    total_time_diffs.append(time_diff)
                    counts.append(1)

        for i in range(len(total_time_diffs)):
            avg_time_differences.append(total_time_diffs[i] / counts[i])

        self.diff_predictions = avg_time_differences # Time difference predictions in seconds


