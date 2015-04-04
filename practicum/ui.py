#!/usr/bin/env python
"""
 AUTHOR: Gabriel Bassett
 DATE: 01-06-2015
 DEPENDENCIES: py2neo, networkx
 Copyright 2014 Gabriel Bassett

 LICENSE:
Copyright 2015 Gabriel Bassett

Licensed under the Apache License, Version 2.0 (the "License") for non-commercial use only;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
To request a commercial license, please contact the author at gabe@infosecanalytics.com

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
TRAIN_RECORDS = 100000  # Number of records to use in training the model.  I'd recommend 100-1000 times the number of diagnoses
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
import operator
import pprint

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
parser.add_argument('--host', help='ip address to use for hosting', default=None)
parser.add_argument('--port', help='port to host the app on', default=None)
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
            HOST = config.get('SERVER', 'host')
        if 'port' in config.options('SERVER'):
            PORT = int(config.get('SERVER', 'port'))
    if config.has_section('MEDICAL'):
        pass # TODO import medical variables and pass to synthetic and model modules
    if config.has_section('MODEL'):
        if 'diagnoses' in config.options('MODEL'):
            DIAGNOSES = int(config.get('MODEL', 'diagnoses'))
        if 'signs' in config.options('MODEL'):
            SIGNS = int(config.get('MODEL', 'signs'))
        if 'symptoms' in config.options('MODEL'):
            SYMPTOMS = int(config.get('MODEL', 'symptoms'))
        if 'training_records' in config.options('MODEL'):
            TRAIN_RECORDS = int(config.get('MODEL', 'training_records'))
        if 'model_dir' in config.options('MODEL'):
            MODEL_DIR = config.get('MODEL', 'model_dir')

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
print "Creating {0} synthetic noisy records.".format(TRAIN_RECORDS)
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
api_parser.add_argument('record_count', type=int, help="The number of records requested.", default=1)
api_parser.add_argument('truth', type=str, help="The name of the diagnosis a truth record is being requested for.", default=None)
#api_parser.add_argument('ASN2', type=str, help="Second ASN of query.  (Order doesn't matter.)", default=None)
#api_parser.add_argument('verizon', type=bool, help="Report on verizon existance in ASN's paths.", default=False)
#api_parser.add_argument('source', type=str, help="An ASN representing the source of the traffic", default=False)
#api_parser.add_argument('destination', type=str, help="An IP address or subnet destination." , default=False)


# Initialize the API class
class records(Resource):
    api_parser = None

    def get(self):
        self.api_parser = api_parser
        api_args = self.api_parser.parse_args(strict=False)
        records = data.create_diagnosis_data(data.truth, api_args.record_count, data.default)
        return records


class truth(Resource):
    api_parser = None

    def get(self):
        self.api_parser = api_parser
        api_args = self.api_parser.parse_args()
        return data.truth.get(api_args.truth, {'error': 'diagnosis does not exist in truth data.'})


# Set up the API
api = Api(app)
api.add_resource(records, '/records/')
api.add_resource(truth, '/truth/')

# Set up the GUI
@app.route("/")
def gui():
    return render_template('index.html')


@app.route('/diagnose/', methods=['GET'])
def diagnose():
    print request.args
    items = request.args.items()
    record = {'signs':{}, 'symptoms':{}}
    try:
        for i in range(len(items)):
            sign_or_symptom = items[i][0].split("_")[0]  # TODO: make this less of a HACK
            #print items
            if sign_or_symptom == 'sign':
                record['signs'][items[i][0]] = float(items[i][1])
            elif sign_or_symptom == 'symptom':
                record['symptoms'][items[i][0]] = float(items[i][1])
    except Exception as e:
        print e

    prediction = mdss.query_nx_model(record)
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)[0:5]
    max_val = prediction[0][1]
    prediction = [[k, "{0}%".format(round((max_val-v)/float(max_val), 7) * -100)] for k, v in prediction]
    print "Returning predictions:"
    pprint.pprint(prediction)
    return jsonify(result=prediction)  # TODO: Parse the data a bit before returning it for peat's sake, and fix the scores



def main():
    logging.info('Beginning main loop.')
    print "Model ready for use."
    app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()    
