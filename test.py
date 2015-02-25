#!/usr/bin/env python
"""
 AUTHOR: Gabriel Bassett
 DATE: <01-23-2015>
 DEPENDENCIES: <a list of modules requiring installation>
 Copyright 2015 Gabriel Bassett

 LICENSE:
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

 DESCRIPTION:
 <A description of the software>

 NOTES:
 <No Notes>

 ISSUES:
 <No Issues>

 TODO:
 <No TODO>

"""

# this code can be used for manually running t
"""
GIT_DIR = "/Users/v685573/OneDrive/Documents/MSIA5243/code/practicum"
NUM_TRAIN_RECORDS = 1000000  # Number of records to use in training the model.  I'd recommend 100-1000 times the number of diagnoses
NUM_TEST_RECORDS = 10000  # Number of records to use to use in testing the model
import imp
# Generate Synthetic Data
print "Loading the synthetic data generation module."
fp, pathname, description = imp.find_module("synthetic", [GIT_DIR])
synthetic = imp.load_module("synthetic", fp, pathname, description)
# Create class instance
print "Creating the synthetic data object 'data' and truth data."
data = synthetic.test_data()
# Create records
print "Creating the synthetic noisy records."
data.records = data.create_diagnosis_data(data.truth, NUM_TRAIN_RECORDS, data.default)


# Train a model based on the synthetic data
print "Loading the model module."
fp, pathname, description = imp.find_module("model", [GIT_DIR])
model = imp.load_module("model", fp, pathname, description)
# Create decision support system object
print "Creating the medical decision support system object 'mdss'."
mdss = model.decision_support_system()
print "Creating the model."
mdss.model = mdss.train_nx_model(data.records)

# Use the model to make diagnoses and see how well it did
print "Generating testing records 'test_records'."
test_records = data.create_diagnosis_data(data.truth, NUM_TEST_RECORDS, data.default)

print "Diagnosing the test records."
truth_diagnoses = dict()
predicted_diagnoses = dict()
for i in range(len(test_records)):
    truth_diagnoses[i] = test_records[i].pop('diagnosis')
    predicted_diagnoses[i] = mdss.query_nx_model(test_records[i])
"""



# PRE-USER SETUP
import logging

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES
GIT_DIR = "/Users/v685573/OneDrive/Documents/MSIA5243/code/practicum"
#NUM_TRAIN_RECORDS = 1000000  # Number of records to use in training the model.  I'd recommend 100-1000 times the number of diagnoses
NUM_TEST_RECORDS = 10000  # Number of records to use to use in testing the model
NUM_TRAIN_RECORDS = [100000, 200000, 400000, 600000, 800000, 1000000, 1300000, 1600000, 2500000, 3200000, 5000000]  # list of training record counts to test at
RESULTS_FILE = "/Users/v685573/OneDrive/Documents/MSIA5243/practicum/results.json"
LOG_LEVEL = logging.DEBUG

########### NOT USER EDITABLE BELOW THIS POINT #################


## IMPORTS


import imp
import operator
import json
import pandas as pd

## SETUP
__author__ = "Gabriel Bassett"
logging.basicConfig(level=LOG_LEVEL)

## EXECUTION




def main():
    logging.info('Beginning main loop.')

    # Generate Synthetic Data
    print "Loading the synthetic data generation module."
    fp, pathname, description = imp.find_module("synthetic", [GIT_DIR])
    synthetic = imp.load_module("synthetic", fp, pathname, description)
    # Create class instance
    print "Creating the synthetic data object 'data' and truth data."
    data = synthetic.test_data()

    print "Loading the model module."
    fp, pathname, description = imp.find_module("model", [GIT_DIR])
    model = imp.load_module("model", fp, pathname, description)

    data.records = list()
    results = dict()

    # Create records
    print "Starting the record creation and testing loop."
    for i in range(len(NUM_TEST_RECORDS)):
        if i == 0:
            record_increment = NUM_TEST_RECORDS[i]
        else:
            record_increment = NUM_TEST_RECORDS[i] - NUM_TEST_RECORDS[i - 1]
        print "Creating {0} more synthetic noisy records for a total of {1}.".format(record_increment, len(data.records) + record_increment)
        data.records = data.records + data.create_diagnosis_data(data.truth, record_increment, data.default)
        # Train a model based on the synthetic data
        # Create decision support system object
        print "Creating the medical decision support system object 'mdss'."
        mdss = model.decision_support_system()
        print "Creating the model."
        mdss.model = mdss.train_nx_model(data.records)

        # Use the model to make diagnoses and see how well it did
        print "Generating {0} testing records 'test_records' for {0} trainging records.".format(NUM_TEST_RECORDS, len(data.records))
        test_records = data.create_diagnosis_data(data.truth, NUM_TEST_RECORDS, data.default)

        print "Diagnosing the test records."
        truth_diagnoses = dict()
        predicted_diagnoses = dict()
        for i in range(len(test_records)):
            truth_diagnoses[i] = test_records[i].pop('diagnosis')
            predicted_diagnoses[i] = mdss.query_nx_model(test_records[i])

        print "Scoring the predictions"
        # Count number in top 1 and 5
        results[NUM_TEST_RECORDS[i]] = {
            'top': 0,
            'top5': 0,
            'locs': [],
            'scores': []
        }
        for i in range(len(predicted_diagnoses)):
            # TODO: Make 'predictions' a matrix or dataframe
            predictions = sorted(predicted_diagnoses[i].items(), key=operator.itemgetter(1), reverse=True)
#            predictions = pd.DataFrame(data={"diagnosis":predicted_diagnoses[i].keys(), "score":predicted_diagnoses[i].values()})
#            predictions.sort(columns='score', ascending=False, inplace=True)
            if predictions[0][0] == truth_diagnoses[i]:
                results[NUM_TRAIN_RECORDS[i]]['top'] += 1
                results[NUM_TRAIN_RECORDS[i]]['top5'] += 1
            elif truth_diagnoses[i] in [key for key, value in predictions[0:5]]:
                results[NUM_TRAIN_RECORDS[i]]['top5'] += 1
            # todo: find location of the diagnosis in the sorted list of diagnoses
            # TODO: predictions.index(truth_diagnoses[i]) needs to be replaced with a real thing
            try:
                loc = predictions.index(truth_diagnoses[i])
                loc = loc + 1 # because values starting at 0 will confuse people
            except ValueError:
                loc = -1
            results[NUM_TRAIN_RECORDS[i]]['locs'].append(loc)
            # find the score difference
            if loc == -1:
                score_diff = 1
            else:
                # TODO: predictions[0]['score'] and predictions[i-1]['score'] need to be replaced with real things
                score_diff = round((predictions[0]['score']-predictions[i-1]['score'])/float(predictions[0]['score']), 7)
            results[NUM_TRAIN_RECORDS[i]]['scores'].append(score_diff)

        print "Writing {0} training record results.".format(len(data.records))
        with open(RESULTS_FILE, 'r') as f:
            json.dump(results, f)

    print "Score the predictions."
    pass # TODO: Notionally going to do this on number of accuracy of top score & accuracy of top 5 scores

    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()