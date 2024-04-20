import cmd
from typing import IO

from time_predictors.baseline_time_predictor import BaselineTimePredictor
from event_predictors.baseline_not_decision_tree import predictor as baselineEventPredictor
from event_predictors.predictor_multiple_columns import predictor as eventPredictor
from time_predictors.predictor_multiple_columns import predictor as timePredictor

from datetime import datetime
import os
import helper_functions
import platform

COLUMNLIST = ['positionInTrace', 'concept:name']

helpText = '''
Commands:

train <path>                                                Train the models on the data in the path variable (if path is not provided default path will be used)
info baseline                                               Get goodness of fit info of the baseline model
info extended                                               Get goodness of fit info of the extended model
dump                                                        (Only for baseline) Dump all combinations of event and event index in trace, and the next event with the probability
predict baseline <index in trace> <timestamp yyyy-mm-dd>    Predict for an index in a trace the most likely event at index + 1 and its timestamp
export                                                      Export a csv of the given data with additional columns containing predictions for the next event and the time. 
events <path>                                          List all events in the data set


help                                                        Display this screen
cls                                                         Clear screen
q                                                           Exit the program
quit                                                        Exit the program

Note: if a dataset has been loaded, no other dataset with the same path can be loaded, the first loaded dataset will be used.
Even if the original dataset is changed after the initial load, the program must be restarted, or the new dataset must be opened with a new path.

Note: The requirements of the dataset are in the README.md file, the libraries required for this script are in the requirements.TXT file.
'''

default_path = "Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
helper = helper_functions.Helper()

class PredictorCLI(cmd.Cmd):

    prompt = '> '
    def __init__(self, completekey: str = "tab", stdin: IO[str] | None = None, stdout: IO[str] | None = None):
        super().__init__(completekey, stdin, stdout)
        # Loading of dataset and construction of models will happen when training
        self.event_predictor = None
        self.time_predictor = None
        self.baseline_event_predictor = None
        self.baseline_time_predictor = None
        self.dataSet = None

        # Mapping model name to info function
        self.info_dict = {
            'baseline': lambda: self.baseline_event_predictor.info(),
            'extended': lambda: self.event_predictor.info()
        }
    def do_train(self, args):
        args = args.split()
        path = args[0] if args else default_path
        self.dataSet = helper.loadData(path)
        print('Fitting time predictor baseline model')
        self.baseline_time_predictor = BaselineTimePredictor(helper.loadData(path))
        print('Fitting event predictor baseline model')
        self.baseline_event_predictor = baselineEventPredictor(helper.loadData(path))
        print('Fitting event predictor multiple columns model')
        self.time_predictor = timePredictor(helper.loadData(path), COLUMNLIST)
        print('Fitting time predictor multiple columns model')
        self.event_predictor = eventPredictor(helper.loadData(path), COLUMNLIST)

    def do_info(self, args):
        args = args.split()
        self.info_dict[args[0]]()

    def do_predict(self, args):
        args = args.split()
        model = args[0]
        if model == 'baseline':
            print(f''' Baseline predicted next event: {self.baseline_event_predictor.predict(int(args[1]))}
            Predicted time of event: {self.baseline_time_predictor.predict
            (datetime.strptime(args[2], '%Y-%m-%d'), int(args[1]))}\n''')
        elif args[0] == 'extended':
            pass  # TODO: Implement prediction for the new model

    def do_export(self, args):
        print('Exporting all predictions, this may take up to a minute...')
        self.dataSet['baseline_predicted_next_time:timestamp'] = self.baseline_time_predictor.predict_all()
        self.dataSet['baseline_predicted_next_concept:name'] = self.baseline_event_predictor.predict_all()
        self.dataSet['multicol_predicted_next_time:timestamp'] = self.time_predictor.predictAll()
        self.dataSet['mutlicol_predicted_next_concept:name'] = self.event_predictor.predictAll()
        self.dataSet.to_csv('predictions.csv', index=False)
        print('Export complete! You can find the results in ./predictions.csv')

    def do_events(self, args):
        args = args.split()
        path = args[0] if args else default_path
        for event in list(helper.loadData(path)['concept:name'].unique()):
            print(event)

    def do_dump(self, args):
        print('Event baseline dump:')
        self.baseline_event_predictor.printProbabilities()
    def do_help(self, arg):
        print(helpText)

    def do_cls(self, args):
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

    def do_quit(self, args):
        print('Exiting...')
        exit()

if __name__ == '__main__':
    try:
        PredictorCLI().cmdloop(intro="Welcome to the process predictor! Type 'help' for a list of commands")
    except Exception as e:
        print(e)
