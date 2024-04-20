import sys
from event_predictors.random_forest import predictor as EventPredictor
from time_predictors.random_forest import predictor as TimePredictor
from helper_functions import Helper

TIMECOLUMNLIST = ['concept:name', 'parallel_cases']
EVENTCOLUMNLIST = ['article', 'amount', 'prefix', 'parallel_cases']

helper = Helper()

if len(sys.argv) > 1:
    print('Loading data file')
    data = helper.loadDataXES(sys.argv[1])
    traces = helper.loadDataXES(sys.argv[2])

else:
    #for demo:
    #data = helper.loadDataPKL().iloc[0:120000]
    #traces = helper.loadDataPKL().iloc[0:120000]
    #for production:
    print('Usage: simple.py path/to/training/data path/to/traces')
    sys.exit(1)

print('Training time predictor')
time_predictor = TimePredictor(data.copy(), TIMECOLUMNLIST)
print('Training event predictor')
event_predictor = EventPredictor(data.copy(), EVENTCOLUMNLIST)

print('Making suffix predictions')
new = event_predictor.predict(traces.copy())
del data
print('Making time predictions')
out = time_predictor.predict_suffix1(new)
del new

out.to_csv('out.csv')