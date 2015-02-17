#!/usr/bin/env python
"""
 AUTHOR: Gabriel Bassett
 DATE: 01-06-2015
 DEPENDENCIES: py2neo, networkx
 Copyright 2014 Gabriel Bassett

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
'''

 DESCRIPTION:
 Implmementation of a graph-based medical decision support system.

"""
# PRE-USER SETUP
import numpy as np

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES

# SET RANDOM SEED 
np.random.seed(5052015)

## TRUTH DATA STATIC VARIABLES
DIAGNOSES = 10000
SIGNS = 3000
SYMPTOMS = 150
SIGNS_PER_DIAG_MEAN = 5.5
SIGNS_PER_DIAG_SD = 0.5  # Increased from 0.25 to give a bit of variance consistent with physician suggestion
SYMPTOMS_PER_DIAG_MEAN = 7 
SYMPTOMS_PER_DIAG_SD = 0.5  # Increased from 0.5 to give a variance consistent with physician suggestion
PERCENT_CONTINUOUS_SIGNS = 0.05
PERCENT_CONTINUOUS_SYMPTOMS = 0.05
PREFERENTIALLY_ATTACH_SIGNS = True
PREFERENTIALLY_ATTACH_SYMPTOMS = False
CONFIG_FILE = "./mdss.cfg"
FLASK_DEBUG = True
HOST = '0.0.0.0'
PORT = 8080
MODEL_DIR = "/Users/v685573/OneDrive/Documents/MSIA5243/code/practicum"
TRAIN_RECORDS = 1000000  # Number of records to use in training the model.  I'd recommend 100-1000 times the number of diagnoses
TEST_RECORDS = 10000  # Number of records to use to use in testing the model

## RECORDS STATIC VARIABLES


########### NOT USER EDITABLE BELOW THIS POINT #################


## IMPORTS
from py2neo import Graph as py2neoGraph
import networkx as nx
import argparse
import logging
from flask import Flask, jsonify, render_template, request
from flask.ext.restful import reqparse, Resource, Api, abort
from collections import defaultdict
import copy
import imp
import ConfigParser

## SETUP
__author__ = "Gabriel Bassett"
# Parse Arguments (should correspond to user variables)
parser = argparse.ArgumentParser(description='This script processes a graph.')
parser.add_argument('-d', '--debug',
                    help='Print lots of debugging statements',
                    action="store_const", dest="loglevel", const=logging.DEBUG,
                    default=logging.WARNING
                   )
parser.add_argument('-v', '--verbose',
                    help='Be verbose',
                    action="store_const", dest="loglevel", const=logging.INFO
                   )
parser.add_argument('--log', help='Location of log file', default=None)
parser.add_argument('host', help='ip address to use for hosting', default=None)
parser.add_argument('port', help='port to host the app on', default=None)
args = parser.parse_args()

# Parse Config File
#print "Config"  # DEBUG
try:
  config = ConfigParser.SafeConfigParser()
  config.readfp(open(CONFIG_FILE))
  config_exists = True
except Exception as e:
    print e #  DEBUG
    config_exists = False
#print "Config is {0}".format(config_exists)
if config_exists:
    if config.has_section('LOGGING'):
        if 'level' in config.options('LOGGING'):
            level = config.get('LOGGING', 'level')
            if level == 'debug':
                loglevel = logging.DEBUG
                FLASK_DEBUG = True
            elif level == 'verbose':
                loglevel = logging.INFO
            else:
                loglevel = logging.WARNING
        else:
            loglevel = logging.WARNING
        if 'log' in config.options('LOGGING'):
            log = config.get('LOGGING', 'log')
            if log.lower() == 'none':
                log = None
        else:
            log = None
    if config.has_section('SERVER'):
        #print 'config arules'  # DEBUG
        if 'host' in config.options('SERVER'):
            #print 'config rules'  # DEBUG
            HOST = config.get('host', 'SERVER')
        if 'port' in config.options('SERVER'):
            PORT = int(config.get('port', 'SERVER'))
    if config.has_section('MEDICAL'):
        pass # TODO import medical variables and pass to synthetic and model modules
    if config.has_section('MODEL'):
        if 'diagnoses' in config.options('MODEL'):
            DIAGNOSES = int(config.get('diagnoses', 'MODEL'))
        if 'signs' in config.options('MODEL'):
            SIGNS = int(config.get('signs', 'MODEL'))
        if 'symptoms' in config.options('MODEL'):
            SYMPTOMS = int(config.get('symptoms', 'MODEL'))
        if 'training_records' in config.options('MODEL'):
            TRAINING_RECORDS = int(config.get('training_records', 'MODEL'))
        if 'model_dir' in config.options('MODEL'):
            MODEL_DIR = config.get('model_dir', 'MODEL')

# Parse Logging Arguments
## Set up Logging
if args.log is not None:
    logging.basicConfig(filename=args.log, level=args.loglevel)
else:
    logging.basicConfig(level=args.loglevel)
if args.loglevel == logging.DEBUG:
    FLASK_DEBUG = True
if args.host is not None:
    HOST = args.host
if args.port is not None:
    PORT = int(args.port)

## EXECUTION
# Set up the mdss model
# Generate Synthetic Data
print "Loading the synthetic data generation module."
fp, pathname, description = imp.find_module("synthetic", [MODEL_DIR])
synthetic = imp.load_module("synthetic", fp, pathname, description)
# Create class instance
print "Creating the synthetic data object 'data' and truth data."
data = synthetic.test_data()
# Create records
print "Creating the synthetic noisy records."
data.records = data.create_diagnosis_data(data.truth, TRAIN_RECORDS, data.default)

# Train a model based on the synthetic data
print "Loading the model module."
fp, pathname, description = imp.find_module("model", [MODEL_DIR])
model = imp.load_module("model", fp, pathname, description)
# Create decision support system object
print "Creating the medical decision support system object 'mdss'."
mdss = model.decision_support_system()
print "Creating the model."
mdss.model = mdss.train_nx_model(data.records)

# Set up the app
app = Flask(__name__)

# define the API
# Initialize the arguements
api_parser = reqparse.RequestParser()
#api_parser.add_argument('ASN1', type=str, help="First ASN of query. (Order doesn't matter.)", default=None)
#api_parser.add_argument('ASN2', type=str, help="Second ASN of query.  (Order doesn't matter.)", default=None)
#api_parser.add_argument('verizon', type=bool, help="Report on verizon existance in ASN's paths.", default=False)
#api_parser.add_argument('source', type=str, help="An ASN representing the source of the traffic", default=False)
#api_parser.add_argument('destination', type=str, help="An IP address or subnet destination." , default=False)


# Initialize the API class
class ui(Resource):
    api_args = None
    g = None

    def __init__(self):
        print "Loading Graph."
        self.api_parser = parser
        # TODO: Load connection to graph
        print "Graph loaded."

    def post(self):
        self.api_args = self.parser.parse_args()
        # ToDO: Parse args and do something


# Set up the API
api = Api(app)
api.add_resource(ui, '/api')


# Set up the GUI
@app.route("/")
def gui():
    return render_template('index.html')


@app.route('/diagnose/', methods=['GET'])
def diagnose():
    print request.args

    record = {}
    for i in range(len(request.args)):
        # TODO: get the output into sign/symptom, name, value
        pass

    ret_data = mdss.query_nx_model(record)
    return jsonify(ret_data)


def main():
    logging.info('Beginning main loop.')

    app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()    
