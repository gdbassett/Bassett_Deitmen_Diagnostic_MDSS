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


 TODO:
  Consider http://docs.scipy.org/doc/scipy/reference/stats.html for all statistical functions

"""
# PRE-USER SETUP
import numpy as np
import scipy

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES
NEODB = "http://192.168.121.134:7474/db/data"

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
SYMPTOMS_PER_CHART_MEAN = 5
SYMPTOMS_PER_CHART_SD = .6
SIGNS_PER_CHART_MEAN = 2.5
SIGNS_PER_CHART_SD = .3


## RECORDS STATIC VARIABLES


########### NOT USER EDITABLE BELOW THIS POINT #################


## IMPORTS
from py2neo import Graph as py2neoGraph
import networkx as nx
import argparse
import logging
from flask import Flask
from flask.ext.restful import reqparse, Resource, Api, abort
from collections import defaultdict
import copy

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
# <add arguments here>
parser.add_argument('db', help='URL of the neo4j graph database', default=NEODB)
#args = parser.parse_args()
## Set up Logging
#if args.log is not None:
#    logging.basicConfig(filename=args.log, level=args.loglevel)
#else:
#    logging.basicConfig(level=args.loglevel)
# <add other setup here>
# Connect to database
G = neo4j.GraphDatabaseService(NEODB)
g = nx.DiGraph()
NEODB = args.db


## EXECUTION
class test_data():
    truth = None
    dists = None

    def __init__(self):
        self.truth = self.create_truth_data()
        self.dists = {
            "bool": self.dist_bool,
            "step_3": self.dist_3_step,
            "step_10": self.dist_10_step,
            "log": self.dist_log,
            "normal": self.dist_normal
        }



    def dist_step(self, x, levels):
        return levels[x-1]


    def dist_bool(self, x, inverse = False):
        if x not in range(1, 3):
            raise ValueError("x must be between 1 and 2")
        if inverse:
            return self.dist_step(x, [1,0])
        else:
            return self.dist__step(x, [0,1])


    def dist_3_step(self, x, levels):
        if x not in range(1, 4):
            raise ValueError("x must be between 1 and 3")
        if len(levels) != 3:
            raise ValueError("levels must have 3 levels")
        if min(levels) < -1 or max(levels) > 1:
            raise ValueError("levels must be confidences between -1 and 1")
        return self.dist_step(x, levels)


    def dist_10_step(self, x, levels):
        if x not in range(1, 11):
            raise ValueError("x must be between 1 and 10")
        if len(levels) != 10:
            raise ValueError("levels must have 10 levels")
        if min(levels) < -1 or max(levels) > 1:
            raise ValueError("levels must be confidences between -1 and 1")
        return self.dist_step(x, levels)


    def dist_log(self, x, k=1, x0=0, pos = True):
        # f -> L as x -> oo (L = vertical width)
        # 4 / k = 98
        # x0 = horizontal centerpoint
        # o = vertical centerpoint
        # If pos = False, sign is flipped on L and o to create a negative relationship
        if pos:
            o = -1
            L = 2
        else:
            o = 1
            L = -2

        return L / (1 + np.exp( -k * (x - x0))) + o


    def dist_normal(self, x, mean=0, sd=1):
        return ((1/(sd * np.sqrt(2 * np.pi)) * np.exp(-((x-mean)**2/(2 * sd)**2))) /
                (1/(sd * np.sqrt(2 * np.pi)) * np.exp(-((mean-mean)**2/(2 * sd)**2))))  #  Normalize so max value is 1
        # TODO: Consider replacing with rv = scipy.stats.norm(loc=0, scale=1) # I think loc=mean and scale=SD
        #         return rv.rvs(size=1)
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm

    def diagnosis_struct(self):
        return {'signs':{}, 'symptoms':{}}

    def create_truth_data(self):
        """
        
        :param signs_symptoms: a list of all potential signs and symptoms
        :param diagnoses: a list of all potential diagnoses
        :param SnS_dist: the median and standard distribution of the number of signs and symptoms in medical lieterature
        :return: a dictionary of {diagnosis: [list of signs and symptoms]} picked probabilistically to create the requested distribution
        """
        # Create Data Sets
        diagnoses = ["diagnosis_{0}".format(x) for x in range(DIAGNOSES)]  # Generate Diagnoses
        signs = ["sign_{0}".format(x) for x in range(SIGNS)]  # Generate Signs
        symptoms = ["symptom_{0}".format(x) for x in range(SYMPTOMS)]  # Generate Symptoms

        # Pick sign/symptom distributions
        distributions = dict()
        continuous_signs = [int(x * len(signs)) for x in np.random.sample(len(signs) * PERCENT_CONTINUOUS_SIGNS)]
        continuous_symptoms = [int(x * len(symptoms)) for x in np.random.sample(len(symptoms) * PERCENT_CONTINUOUS_SYMPTOMS)]
        categorical_functions = ['bool', 'step_3', 'step_10']
        continuous_functions = ['log', 'normal']
        for sign in signs:
            if sign in continuous_signs:
                distributions[sign] = continuous_functions[int(np.random.sample() * 2)]
            else:
                distributions[sign] = categorical_functions[int(np.random.sample() * 3)]
        for symptom in symptoms:
            if symptom in continuous_symptoms:
                distributions[symptom] = continuous_functions[int(np.random.sample() * 2)]
            else:
                distributions[symptom] = categorical_functions[int(np.random.sample() * 3)]
        # Clean up
        del(continuous_signs)
        del(continuous_symptoms)
        del(categorical_functions)
        del(continuous_functions)

        # Create sign/symptom to diagnosis relationship
        truth = defaultdict(self.diagnosis_struct)
        signs_preferential = copy.deepcopy(signs)
        symptoms_preferential = copy.deepcopy(symptoms)
        signs_per_diag_set = list(np.random.normal(SIGNS_PER_DIAG_MEAN, SIGNS_PER_DIAG_SD, DIAGNOSES))  # Generate the set of count of signs per diagnosis
        symptoms_per_diag_set = list(np.random.normal(SYMPTOMS_PER_DIAG_MEAN, SYMPTOMS_PER_DIAG_SD, DIAGNOSES))  # Generate the set of count of symptoms per diagnosis
        for diagnosis in diagnoses:
            if PREFERENTIALLY_ATTACH_SIGNS:
                for sign in range(int(signs_per_diag_set.pop())):
                    # Choose a sign from the constantly updated preferential list
                    s = signs_preferential[int(np.random.sample() * len(signs_preferential))]
                    truth[diagnosis]['signs'][s] = {'function': distributions[s], 'factors':{}}
                    # Add the sign to the preferential list so that the it is more likely to be chosen next time.
                    signs_preferential.append(s)
            else:
                for sign in range(int(signs_per_diag_set.pop())):
                    s = signs[int(np.random.sample() * len(signs))]
                    truth[diagnosis]['signs'][s] = {'function': distributions[s], 'factors':{}}

            if PREFERENTIALLY_ATTACH_SYMPTOMS:
                for symptom in range(int(symptoms_per_diag_set.pop())):
                    s = symptoms_preferential[int(np.random.sample() * len(symptoms_preferential))]
                    truth[diagnosis]['symptoms'][s] = {'function': distributions[s], 'factors':{}}
                    symptoms_preferential.append(s)
            else:
                for symptom in range(int(symptoms_per_diag_set.pop())):
                    # randomly choose a symptom and append it to the symptoms list
                    s = symptoms[int(np.random.sample() * len(symptoms))]
                    truth[diagnosis]['symptoms'][s] = {'function': distributions[s], 'factors':{}}

        # clean up of variables which shouldn't be trusted
        del(signs_per_diag_set)
        del(symptoms_per_diag_set)
        del(signs_preferential)
        del(symptoms_preferential)

        # Assign Distribution Characteristics to Diagnosis-Symptom
        ## Distribution characteristics are limited to positive or negative relationships to features
        for diagnosis in truth.keys():
            for sign in truth[diagnosis]:
                function =  truth[diagnosis]['signs'][sign]['function']
                if function == 'bool':
                    if np.random.binomial(1, .5):
                        factors = {'inverse': False}
                    else:
                        factors = {'inverse': True}
                    f_type = 'categorical'
                elif function == 'step_3':
                    if np.random.binomial(1, .5):
                        factors = {'levels': [-1, .5, 1]}
                    else:
                        factors = {'levels': [1, .5, -1]}
                    f_type = 'categorical'
                elif function == 'step_10':
                    if np.random.binomial(1, .5):
                        factors = {'levels': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]}
                    else:
                        factors = {'levels': [1, .9, .8, .7, .6, .5, .4, .3, .2, .1]}
                    f_type = 'categorical'
                elif function == 'log':
                    if np.random.binomial(1, .5):
                        factors = {'pos': True}
                    else:
                        factors = {'pos': False}
                    f_type = 'continuous'
                elif function == 'normal':
                    # Randomly choose a mean between -1 and 1 and SD between 0.2 and 1
                    factors = {'mean': np.random.sample() * 2 - 1, 'sd': np.random.sample() * .8 + .2}
                    f_type = 'continuous'
                else:
                    raise KeyError("Sign Function not found in functions list.")
                truth[diagnosis]['signs'][sign]['factors'] = factors
                truth[diagnosis]['signs'][sign]['function_type'] = f_type
            for symptom in truth[diagnosis]:
                function =  truth[diagnosis]['symptoms'][symptom]['function']
                if function == 'bool':
                    if np.random.binomial(1, .5):
                        factors = {'inverse': False}
                    else:
                        factors = {'inverse': True}
                    f_type = 'categorical'
                elif function == 'step_3':
                    if np.random.binomial(1, .5):
                        factors = {'levels': [-1, .5, 1]}
                    else:
                        factors = {'levels': [1, .5, -1]}
                    f_type = 'categorical'
                elif function == 'step_10':
                    if np.random.binomial(1, .5):
                        factors = {'levels': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]}
                    else:
                        factors = {'levels': [1, .9, .8, .7, .6, .5, .4, .3, .2, .1]}
                    f_type = 'categorical'
                elif function == 'log':
                    if np.random.binomial(1, .5):
                        factors = {'pos': True}
                    else:
                        factors = {'pos': False}
                    f_type = 'continuous'
                elif function == 'normal':
                    # Randomly choose a mean between -1 and 1 and SD between 0.2 and 1
                    factors = {'mean': np.random.sample() * 2 - 1, 'sd': np.random.sample() * .8 + .2}
                    f_type = 'continuous'
                else:
                    raise KeyError("Sign Function not found in functions list.")
                truth[diagnosis]['symptoms'][symptom]['factors'] = factors
                truth[diagnosis]['symptoms'][symptom]['function_type'] = f_type

        return truth

# noinspection PyUnreachableCode
def create_diagnosis_data(self, truth_data, records, SnS_dist = (SNSmedian, SNSSD), pct_true_sign=.99, pct_true_symptom=.95):
        """

        :param truth data: a dictionary of {diagnosis: [list of signs and symptoms]} representing ground truth
        :param records: integer representing the number of records to generate
        :param SnS_dist: the median and standard distribution of the number of signs and symptoms in a normal chart
        :param pct_true_sign: float representing the percentage of signs which will be from those associated with the diagnosis
        :param pct_true_symptom: float representing the percentage of symptoms which will be from those associated with the diagnosis
        :return: a dictionary of {diagnosis: [list of signs and symptoms]} picked probabilistically to create the requested distribution

        NOTE: The returned dictionary will pick signs and symptoms so that false signs/symptoms are outliers, but will add false positives
        NOTE: The returned dictionary will use preferential attachment for both true and false positive signs and symptoms.
        """
        # Generate the baseline function used to decide whether to add a true or a false sign/symptom to the record
        baseline = scipy.stats.halfnorm(loc=0, scale=1)  # mean of 0, SD of 1  call with size=X to return X values that fit the distribution

        # Create the records holder
        synthetic_records = list()

        # Convert the percent true to a cutoff.
        cutoff_sign = baseline.ppf(pct_true_sign)
        cutoff_symptom = baseline.ppf(pct_true_symptom)

        # Generate the default diagnosis from the truth data.  This will be used for the factors for false signs/symptoms
        pass # TODO


        for record in range(records):
            # Choose a diagnosis from the truth data
            diagnosis = np.random.choice(truth_data.keys())

            # choose a number of symptoms based on marcus's numbers
            num_symptoms = round(scipy.stats.norm.rvs(loc=SYMPTOMS_PER_CHART_MEAN, scale=SYMPTOMS_PER_CHART_SD))

            # choose a number of signs based marcus's numbers
            num_signs = round(scipy.stats.norm.rvs(loc=SIGNS_PER_CHART_MEAN, scale=SIGNS_PER_CHART_SD))
        
            for sign in num_signs:
                if sign in 'continuous':  # TODO
                    # pick a random number
                    pass  # TODO
                    # pick a value out of the distribution for the sign based on the random number
                    pass  # TODO
                
                else:  # sign is
                    # choose if true or false
                    # pick a random number.  If it is above the cutoff, pick a random false sign w/ the default diagnosis factors
                    #  If it is at or below the cutoff, pick a true sign/symptom
                    #    Pick the level with a normal distribution weighted towards the true side
                    pass  # TODO
                # Store the sign and value in the record
                pass  # TODO
            
            for symptom in num_symptoms:
                if symptom in 'continuous':  # TODO
                    # pick a random number
                    pass  # TODO
                    # pick a value out of the distribution for the symptom based on the random number
                    pass  # TODO
                
                else:  # symptom is
                    # choose if true or false
                    # pick a random number.  If it is above the cutoff, pick a random false symptom w/ the default diagnosis factors
                    #  If it is at or below the cutoff, pick a true symptom/symptom
                    #    Pick the level with a normal distribution weighted towards the true side
                    pass  # TODO
                # Store the symptom and value in the record
                pass  # TODO

            # Store record to list of records
            pass # todo

        return synthetic_records


def main():
    logging.info('Beginning main loop.')

    # Initialize the arguements
    api_parser = reqparse.RequestParser()
    #api_parser.add_argument('ASN1', type=str, help="First ASN of query. (Order doesn't matter.)", default=None)
    #api_parser.add_argument('ASN2', type=str, help="Second ASN of query.  (Order doesn't matter.)", default=None)
    #api_parser.add_argument('verizon', type=bool, help="Report on verizon existance in ASN's paths.", default=False)
    #api_parser.add_argument('source', type=str, help="An ASN representing the source of the traffic", default=False)
    #api_parser.add_argument('destination', type=str, help="An IP address or subnet destination." , default=False)

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(ASNSearch, '/')
    app.run(debug=True)
    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()    
