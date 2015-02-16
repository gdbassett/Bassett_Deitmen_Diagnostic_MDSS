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
# PRE-USER SETUP
pass

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES
NEODB = "http://192.168.121.134:7474/db/data"
CONFIG_FILE = "/tmp/verum.cfg"
GIT_DIR = "/Users/v685573/OneDrive/Documents/MSIA5243/code/practicum"
NUM_TRAIN_RECORDS = 1000000  # Number of records to use in training the model.  I'd recommend 100-1000 times the number of diagnoses
NUM_TEST_RECORDS = 10000  # Number of records to use to use in testing the model

########### NOT USER EDITABLE BELOW THIS POINT #################


## IMPORTS
from py2neo import neo4j, cypher
import networkx as nx
import argparse
import logging
import ConfigParser
import imp

## SETUP
__author__ = "Gabriel Bassett"
# Parse Arguments (should correspond to user variables)
parser = argparse.ArgumentParser(description='This script processes a graph.')
parser.add_argument('-d', '--debug',
                    help='Print lots of debugging statements',
                    action="store_const", dest="loglevel", const=logging.DEBUG,
                    default=loglevel
                   )
parser.add_argument('-v', '--verbose',
                    help='Be verbose',
                    action="store_const", dest="loglevel", const=logging.INFO
                   )
parser.add_argument('--log', help='Location of log file', default=log)
parser.add_argument('--config', help='The location of the config file', default=CONFIG_FILE)
# <add arguments here>
parser.add_argument('--db', help='URL of the neo4j graph database', default=NEODB)
args = parser.parse_args()
# Set up config file
CONFIG_FILE = args.config
try:
  config = ConfigParser.SafeConfigParser()
  config.readfp(open(CONFIG_FILE))
  config_exists = True
except:
  config_exists = False
if config_exists:
    if config.has_section('LOGGING'):
        if 'level' in config.options('LOGGING'):
            level = config.get('LOGGING', 'level')
            if level == 'debug':
                loglevel = logging.DEBUG
            elif level == 'verbose':
                loglevel = logging.INFO
            else:
                loglevel = logging.WARNING
        else:
            loglevel = logging.WARNING
        if 'log' in config.options('LOGGING'):
            log = config.get('LOGGING', 'log')
        else:
            log = None
    if config.has_section('NEO4J'):
        if 'db' in config.options('NEO4J'):
            NEODB = config.get('NEO4J', 'db')
# <add additional config options here>


## Set up Logging
if args.log is not None:
    logging.basicConfig(filename=args.log, level=args.loglevel)
else:
    logging.basicConfig(level=args.loglevel)
# <add other setup here>
# Connect to database
G = neo4j.GraphDatabaseService(NEODB)
g = nx.DiGraph()
NEODB = args.db


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

    print "Score the predictions."
    pass # TODO: Notionally going to do this on number of accuracy of top score & accuracy of top 5 scores

    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()