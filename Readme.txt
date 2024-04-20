(This is my project with 6 people, we creating and training a model to predict the future event and time of that event)

About the project:

With the provided data set from Road Traffic Fine Management Process, we write a model to predict name and time
of upcoming event. We used python, two main libraries pm4py and pandas to analyse and make prediction.

Getting Started:
The tool has been tested and works on Python 3.10 and above. To run the commands on Windows you can use the command
prompt or powershell, on Linux and MacOS use terminal.

1. Open the appropriate shell for you operating system in the 'Process-mining' directory.

2. Install the required libraries by running

'pip install -r requirements.txt'

3. Run the simple.py script

'python3 simple.py PATH/TO/DATASET.xes PATH/TO/TRACE.xes'

After a few moments (depending on the size of the dataset), a file named out.csv will be generated containing the
suffix predictions for the traces in the trace file.

Requirements for the predictor:

For the tool to work, the following columns mnust be present in the datset

- A column called 'time:timestamp' containing the time at which the event occured in timestamp format.
- A column called 'case:concept_name' containing the name of the case.
- A column called 'concept:name' containing the name of the event.
- A column called '@@case_index' containing the unique id of the case.
- a column called '@@index' containing the numerical unique id of the event this column must have the property that all
- a column called 'org:resource' which contains a unique id for the person handling the case
ids are sequential when the df is ordered by '@@case_index' and that the spacing between each value is the same. i.e.
1 - 2 - 3, or 0.5, 1, 1.5. No values must be skipped either.
The dataset must primarily be ordered by @@case_index, and then by @@index
Each case must start with the 'Create Fine' event
