#!/usr/bin/env python
"""
 AUTHOR: Gabriel Bassett
 DATE: 01-06-2015
 DEPENDENCIES: py2neo, networkx
 Copyright 2014 Gabriel Bassett

 LICENSE:
 This software is not licenced for use by those other than the author.
'''

 DESCRIPTION:
 Implmementation of a graph-based medical decision support system.


 TODO:
  -Consider http://docs.scipy.org/doc/scipy/reference/stats.html for all statistical functions
  -Add provide a continuous linear distribution for signs/symptoms in addition to normal and cumulative density (referred to as log in code)

"""
# PRE-USER SETUP
import numpy as np
import scipy.stats

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES
NEODB = "http://192.168.121.134:7474/db/data"

# SET RANDOM SEED 
np.random.seed(5052015)

## TRUTH DATA STATIC VARIABLES
## Based on consultation with domain specialist, (ER MD, Acting medical director of hospital, 30 years experience, etc.
DIAGNOSES = 10000
SIGNS = 3000
SYMPTOMS = 150
TESTS = SIGNS  # There is 1 test for each sign
MAX_TREATED = 100  # The maximum number of diagnoses a treatment will treat
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
PCT_FALSE_SIGNS = .08
PCT_FALSE_SYMPTOMS = .2


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
#G = neo4j.GraphDatabaseService(NEODB)
#g = nx.DiGraph()
#NEODB = args.db


## EXECUTION
class test_data():
    truth = None
    dists = None
    default = None
    records = None

    def __init__(self):
        self.truth, self.default = self.create_truth_data()
        self.dists = {
            "bool": self.dist_bool,
            "step_3": self.dist_3_step,
            "step_10": self.dist_10_step,
            "log": self.dist_log,
            "normal": self.dist_normal
        }


    def create_diagnosis_data(self, truth_data, records, default_diagnosis, pct_true_sign=.92, pct_true_symptom=.80):
        """

        :param truth data: a dictionary of {diagnosis: [list of signs and symptoms]} representing ground truth
        :param records: integer representing the number of records to generate
        :param pct_true_sign: float representing the percentage of signs which will be from those associated with the diagnosis.
        :param pct_true_symptom: float representing the percentage of symptoms which will be from those associated with the diagnosis
        :return: a dictionary of {diagnosis: [list of signs and symptoms]} picked probabilistically to create the requested distribution

        NOTE: The returned dictionary will pick signs and symptoms so that false signs/symptoms are outliers, but will
                add false positives
        NOTE: pct_true_sign/pct_true_symptom choose both the number of true/false signs/symptoms, but also, the
                likelihood that boolean signs/symptoms will be true/false
        NOTE: Non-boolean signs/symptoms will be chosen over a half-normal or normal distribution with the mean set at
               the correct value & 3SD set at the bottom of the range of potential values, (or not set at all for the
               actual normally distributed signs/symptoms)
        NOTE: The returned dictionary will use preferential attachment for both true and false positive signs and
            symptoms.
        NOTE: Because generation of false signs/symptoms from the default diagnosis does not check if the sign/symptom
               is an actual sign/symptom of the diagnosis, it's possible for signs/symptoms from the false branch to
               overwrite those from the true branch and visa versa.  In practice this shouldn't be an issue as it is
               low likelihood and should do nothing more than slightly effect the distributions of true/false
               signs/symptoms and their values
        """
        # Generate the baseline function used to decide whether to add a true or a false sign/symptom to the record
        mean = 0
        SD = 1
        baseline = scipy.stats.halfnorm(loc=mean, scale=SD)  # mean of 0, SD of 1  call with size=X to return X values that fit the distribution

        # Create the records holder
        synthetic_records = list()

        # Convert the percent true to a cutoff.
        cutoff_sign = baseline.ppf(pct_true_sign)
        cutoff_symptom = baseline.ppf(pct_true_symptom)

        for i in range(records):
            record = self.diagnosis_struct()

            # Choose a diagnosis from the truth data
            record['diagnosis'] = np.random.choice(truth_data.keys())

            # choose a number of symptoms based on marcus's numbers
            num_symptoms = int(round(scipy.stats.norm.rvs(loc=SYMPTOMS_PER_CHART_MEAN, scale=SYMPTOMS_PER_CHART_SD)))

            # choose a number of signs based marcus's numbers
            num_signs = int(round(scipy.stats.norm.rvs(loc=SIGNS_PER_CHART_MEAN, scale=SIGNS_PER_CHART_SD)))

            for j in range(num_signs):
                # If a random number is below the cutoff, choose a correct sign
                #  Second qualification is to ensure we don't duplicate true signs
                # TODO: Logic below is incorrect as len(record['signs']) could count false signs in addition to true ones
                if baseline.rvs() < cutoff_sign and len(record['signs']) < len(truth_data[record['diagnosis']]['signs']):
                    try:
                        sign, val = self.get_sign_or_symptom_value(truth_data[record['diagnosis']],
                                                               'sign',
                                                               cutoff_sign,
                                                               baseline,
                                                               mean,
                                                               SD)
                    except:
                        print truth_data[record['diagnosis']]
                        print 'sign'
                        print cutoff_sign
                        print type(baseline)
                        print mean
                        print SD
                        print num_signs
                        raise
                else:
                    sign, val = self.get_sign_or_symptom_value(default_diagnosis,
                                                               'sign',
                                                               cutoff_sign,
                                                               baseline,
                                                               mean,
                                                               SD)
                record['signs'][sign] = val

            for j in range(num_symptoms):
                # TODO: Logic below is incorrect as len(record['symptoms']) could count false symptoms in addition to true ones
                if baseline.rvs() < cutoff_symptom and len(record['symptoms']) < len(truth_data[record['diagnosis']]['symptoms']):
                    symptom, val = self.get_sign_or_symptom_value(truth_data[record['diagnosis']],
                                                                  'symptom',
                                                                  cutoff_symptom,
                                                                  baseline,
                                                                  mean,
                                                                  SD)
                else:
                    symptom, val = self.get_sign_or_symptom_value(default_diagnosis,
                                                                  'symptom',
                                                                  cutoff_symptom,
                                                                  baseline,
                                                                  mean,
                                                                  SD)
                record['symptoms'][symptom] = val

            synthetic_records.append(record)

        return synthetic_records


    def create_truth_data(self, default_diagnosis=True):
        """

        :param signs_symptoms: a list of all potential signs and symptoms
        :param diagnoses: a list of all potential diagnoses
        :param SnS_dist: the median and standard distribution of the number of signs and symptoms in medical lieterature
        :return: a dictionary of {diagnosis: [list of signs and symptoms]} picked probabilistically to create the requested distribution
        """
        #TODO: Return signs, symptoms, and diagnoses and place in the class

        # Create Data Sets
        diagnoses = ["diagnosis_{0}".format(x) for x in range(DIAGNOSES)]  # Generate Diagnoses
        signs = ["sign_{0}".format(x) for x in range(SIGNS)]  # Generate Signs
        symptoms = ["symptom_{0}".format(x) for x in range(SYMPTOMS)]  # Generate Symptoms

        # Pick sign/symptom distributions
        distributions = dict()
#        continuous_signs = [int(x * len(signs)) for x in np.random.sample(len(signs) * PERCENT_CONTINUOUS_SIGNS)]
        continuous_signs = np.random.choice(signs, int(len(signs) * PERCENT_CONTINUOUS_SIGNS))
        #print continuous_signs  # DEBUG
#        continuous_symptoms = [int(x * len(symptoms)) for x in np.random.sample(len(symptoms) * PERCENT_CONTINUOUS_SYMPTOMS)]
        continuous_symptoms = np.random.choice(symptoms, int(len(symptoms) * PERCENT_CONTINUOUS_SYMPTOMS))
        #print continuous_symptoms  # DEBUG
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
                    truth[diagnosis]['signs'][s] = {'function': distributions[s], 'factors':{}, 'function_type':{}}
                    # Add the sign to the preferential list so that the it is more likely to be chosen next time.
                    signs_preferential.append(s)
            else:
                for sign in range(int(signs_per_diag_set.pop())):
                    s = signs[int(np.random.sample() * len(signs))]
                    truth[diagnosis]['signs'][s] = {'function': distributions[s], 'factors':{}, 'function_type':{}}

            if PREFERENTIALLY_ATTACH_SYMPTOMS:
                for symptom in range(int(symptoms_per_diag_set.pop())):
                    s = symptoms_preferential[int(np.random.sample() * len(symptoms_preferential))]
                    truth[diagnosis]['symptoms'][s] = {'function': distributions[s], 'factors':{}, 'function_type':{}}
                    symptoms_preferential.append(s)
            else:
                for symptom in range(int(symptoms_per_diag_set.pop())):
                    # randomly choose a symptom and append it to the symptoms list
                    s = symptoms[int(np.random.sample() * len(symptoms))]
                    truth[diagnosis]['symptoms'][s] = {'function': distributions[s], 'factors':{}, 'function_type':{}}

        # clean up of variables which shouldn't be trusted
        del(signs_per_diag_set)
        del(symptoms_per_diag_set)
        del(signs_preferential)
        del(symptoms_preferential)

        # Assign Distribution Characteristics to Diagnosis-Symptom
        ## Distribution characteristics are limited to positive or negative relationships to features
        for diagnosis in truth.keys():
            for sign in truth[diagnosis]['signs']:
                function =  truth[diagnosis]['signs'][sign]['function']
                factors, f_type = self.get_factors_and_type(function)
                truth[diagnosis]['signs'][sign]['factors'] = factors
                truth[diagnosis]['signs'][sign]['function_type'] = f_type

            for symptom in truth[diagnosis]['symptoms']:
                function =  truth[diagnosis]['symptoms'][symptom]['function']
                factors, f_type = self.get_factors_and_type(function)
                truth[diagnosis]['symptoms'][symptom]['factors'] = factors
                truth[diagnosis]['symptoms'][symptom]['function_type'] = f_type

        # If 'default_diagnosis' is set, create and return a diagnosis that includes all signs and symptoms
        if default_diagnosis:
            default = self.diagnosis_struct()
            for sign in signs:
                function =  distributions[sign]
                factors, f_type = self.get_factors_and_type(function)
                default['signs'][sign] = {
                    'factors': factors,
                    'function_type': f_type,
                    'function': function
                }
            for symptom in symptoms:
                function =  distributions[symptom]
                factors, f_type = self.get_factors_and_type(function)
                default['symptoms'][symptom] = {
                    'factors': factors,
                    'function_type': f_type,
                    'function': function
                }

        # Return
        if default_diagnosis:
            return truth, default
        else:
            return truth


    def create_treatments(self, truth_data=None):
        """

        :return: a dictionary with keys of treatments and values of dictionaries keyed with diagnoses, signs, and symptoms and values of type and impact
        """
         # If truth_data is None, default to truth data in module
        if truth_data is None:
            truth_data = self.truth
        if truth_data is None:
            raise ValueError("Truth data must either be passed to the function or exist in the object.")

        # get list of signs, symptoms, and diagnoses
        signs = set()
        symptoms = set()
        diagnoses = truth_data.keys()
        for diagnosis in truth_data.values():
            signs = signs.add(diagnosis['signs'].keys())
            symptoms = symptoms.add(diagnosis['symptoms'].keys())
        signs = list(signs)
        symptoms = list(symptoms)

        treatments = defaultdict(dict)

        #  1. Create a fair coin
        coin = scipy.stats.bernoulli(.5)
        # Create a coin that comes up true 1/5 of the time
        coin_fifth = scipy.stats.bernoulli(.8)


        # Create treatments for diagnoses
        # Create diagnoses distribution
        # The below beta prime distribution creates a distribution where a fair number (~23% of diagnoses have
        #  no treatment and a small percent (~2%) have more than 10 treatments)
        dist = scipy.stats.betaprime(5, 3)
        #  0. Create a list of used treatments
        used_treatments = list()
        # pick treatments for the diagnoses
        for diagnosis in diagnoses:
            type = 'diagnosis'
            # pick a number of treatments
            treatments_per = int(dist.rvs())
            for treatment in treatments_per:
                # prefer treatments treat a single diagnosis/sign/symptom, otherwise preferentially attach to treat lots of things
                # 2. Flip it
                if len(used_treatments) > 0 and coin.rvs():
                    # use an existing treatment
                    t = np.random.choice(used_treatments)
                else:
                    # Create a new treatment
                    t = "treatment_{0}".format(len(set(used_treatments)))
                used_treatments.append(t)

                #Give the treatment an impact score (either 'cures' or a normal distribution of mean=1, sd = 1/3
                impact_dist = scipy.stats.norm(1, 1/float(3))
                if coin_fifth.rvs():
                    impact = 'max'
                else:
                    impact = impact_dist.rvs()
                # store the treatment
                treatments[t][diagnosis] = {'type':'diagnosis', 'impact':impact}

        # Create treatments for signs
        # Create signs distribution
        # The below lognormal distribution starts at 1 and roughly goes down to 10
        dist = scipy.stats.lognorm(.95, loc=1)
        #  0. Create a list of used treatments
        used_treatments = list()
        # pick treatments for the signs
        for sign in signs:
            # pick a number of treatments
            treatments_per = int(dist.rvs())
            # we don't want anything over 10
            if treatments_per > 10:
                treatments_per = 1
            for treatment in treatments_per:
                # prefer treatments treat a single diagnosis/sign/symptom, otherwise preferentially attach to treat lots of things
                # 2. Flip it
                if len(used_treatments) > 0 and coin.rvs():
                    # use an existing treatment
                    t = np.random.choice(used_treatments)
                else:
                    # Create a new treatment
                    t = "treatment_{0}".format(len(set(used_treatments)))
                used_treatments.append(t)

                #Give the treatment an impact score (either 'cures' or a normal distribution of mean=1, sd = 1/3
                impact_dist = scipy.stats.norm(1, 1/float(3))
                if coin_fifth.rvs():
                    impact = 'max'
                else:
                    impact = impact_dist.rvs()
                # store the treatment
                treatments[t][sign] = {'type':'sign', 'impact':impact}

        # Create treatments for symptoms
        # Create symptoms distribution
        # Use a gamma distribution to get a long tail (out to about a max of 100)
        dist = scipy.stats.gamma(.008, scale=20, loc=2)  # Produces a long tail
        # if the value is below 5, use a normal scale with a mean of 2
        dist2 = scipy.stats.norm(2.5, scale=.5)
        #  0. Create a list of used treatments
        used_treatments = list()
        # pick treatments for the symptoms
        for symptom in symptoms:
            # Pick a number of treatments
            treatments_per = int(dist.rvs())
            if treatments_per < 5:
                treatments_per = int(dist2.rvs())
            if treatments_per < 0:
                treatments_per = 0
            for treatment in treatments_per:
                # prefer treatments treat a single diagnosis/sign/symptom, otherwise preferentially attach to treat lots of things
                # 2. Flip it
                if len(used_treatments) > 0 and coin.rvs():
                    # use an existing treatment
                    t = np.random.choice(used_treatments)
                else:
                    # Create a new treatment
                    t = "treatment_{0}".format(len(set(used_treatments)))
                used_treatments.append(t)

                #Give the treatment an impact score (either 'cures' or a normal distribution of mean=1, sd = 1/3
                impact_dist = scipy.stats.norm(1, 1/float(3))
                if coin_fifth.rvs():
                    impact = 'max'
                else:
                    impact = impact_dist.rvs()
                # store the treatment
                treatments[t][symptom] = {'type':'symptom', 'impact':impact}

        # return the dictionary of treatment-diagnosis/sign/symptom mappings with impact scores
        return treatments


    def create_tests(self, truth_data=None):
        """

        :return: dictionary of dictionaries with keys of tests and values of dictionaries with keys of signs and values including confidence

        NOTE: Supports more than 1 sign per test as well as other features by using dictionary of dictionaries
        """
        # If truth_data is None, default to truth data in module
        if truth_data is None:
            truth_data = self.truth
        if truth_data is None:
            raise ValueError("Truth data must either be passed to the function or exist in the object.")

        # get list of signs from diagnoses
        signs = set()
        for diagnosis in truth_data.values():
            signs = signs.add(diagnosis['signs'].keys())
        signs = list(signs)

        # randomize signs
        np.random.shuffle(signs)

        tests = dict()
        for i in range(TESTS):
            if i >= len(signs):
                break
            # confidence is picked from an exponential distribution normalized to 1 to 0 (at 5-9's)
            confidence = abs((10-scipy.stats.expon.rvs())/10)
            # Create one test per sign and return the linkage
            tests["test_{0}".format(i)] = {signs[i]:{'confidence':confidence}}

        return tests


    def dist_3_step(self, x, levels):
        if x not in range(1, 4):
            raise ValueError("x must be between 1 and 3")
        if len(levels) != 3:
            raise ValueError("levels must have 3 levels")
        if min(levels) < -1 or max(levels) > 1:
            raise ValueError("levels must be confidences between -1 and 1")
        return self.dist_step(x, levels)


    def diagnosis_struct(self):
        return {'signs':{}, 'symptoms':{}}


    def dist_10_step(self, x, levels):
        if x not in range(1, 11):
            raise ValueError("x must be between 1 and 10")
        if len(levels) != 10:
            raise ValueError("levels must have 10 levels")
        if min(levels) < -1 or max(levels) > 1:
            raise ValueError("levels must be confidences between -1 and 1")
        return self.dist_step(x, levels)


    def dist_bool(self, x, inverse = False):
        if x not in range(1, 3):
            raise ValueError("x must be between 1 and 2")
        if inverse:
            return self.dist_step(x, [1,0])
        else:
            return self.dist__step(x, [0,1])


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


    def dist_step(self, x, levels):
        return levels[x-1]


    def generate_records(self, record_count=1000000, pct_tru_sign=.92, pct_true_symptom=.80):
        self.records = self.create_diagnosis_data(self, self.truth,
                                                  record_count,
                                                  self.default,
                                                  pct_tru_sign,
                                                  pct_true_symptom)


    def get_factors_and_type(self, function):
        """

        :param self:
        :param function: a type of distribution used for signs/symptoms
        :return: factors for that distribution and the type of distribution (continuous or categorical)
        """
        if function == 'bool':
            if np.random.binomial(1, .5):
                factors = {'inverse': False}
            else:
                factors = {'inverse': True}
            f_type = 'categorical'
        elif function == 'step_3':
        #    if np.random.binomial(1, .5):
        #        factors = {'levels': [-1, .5, 1]}
        #    else:
        #        factors = {'levels': [1, .5, -1]}
            factors = {'levels': [1, .5, -1]}  # replaces above section.  Deeper issue but base is above produces negative relationships which aren't handled correctly in truth data.
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
            raise KeyError("Function not found in functions list.")

        return factors, f_type


    def get_sign_or_symptom_value(self, diagnosis, sign_or_symptom, cutoff, baseline=None, mean=0, SD=1):
        """

        :param diagnosis: a truth-data diagnosis
        :param sign_or_symptom: Either "sign" or "symptom"
        :param cutoff: the cutoff to use for outliers (generally for false positives)
        :param baseline: a scipy.stats halfnorm frozen distribution  used heavily for picking values
        :param mean: the location of the halfnorm max value (normally 0)
        :param SD: the scale of the halfnorm distribution (normally 1)
        :return: a sign or symptom from the diagnosis and a value for it
        """
        # Allow for some default properties of baseline, mean, and SD
        if baseline is None:
            baseline = scipy.stats.halfnorm(loc=mean, scale=SD)

        if sign_or_symptom is "sign":
            s = np.random.choice(diagnosis['signs'].keys())
        elif sign_or_symptom is 'symptom':
            s = np.random.choice(diagnosis['symptoms'].keys())
        else:
            raise ValueError("sign_or_symptom must be 'sign' or 'symptom'")
        try:
            factors = diagnosis[sign_or_symptom + "s"][s]['factors']
        except:
            print diagnosis
            print sign_or_symptom + "s"
            print s

        # if the sign is normal
        if diagnosis[sign_or_symptom + "s"][s]['function'] == 'normal':
            val = scipy.stats.norm(loc=factors['mean'], scale=factors['sd']).rvs()  # TODO: This should be the normal distribution. do we need to scale to the cutoff for synthetic records?
#            diagnosis[sign_or_symptom + "s"][s] = val
        elif diagnosis[sign_or_symptom + "s"][s]['function'] == 'log':
            # LOG is really the cumulative density function of the norm.

            if diagnosis[sign_or_symptom + "s"][s]['factors']['pos']:
                # the mean of the halfnorm needs to be at 3 (3 SD) and the tail needs to be negative
                # we are going to take a halfnorm dist.  For positive, we'll center (start) at 3SD (3) and
                #   _subtract_ the dist from the mean (3) giving us a reverse distribution
                #  We add 2 * 3SD where SD = 1 to move the start of the half-norm to +3SD going backwards
                intermediate_val = 6*SD - scipy.stats.halfnorm(loc=3*SD).rvs() # (to recenter at the top of the CDF)
                val = scipy.stats.norm().cdf(intermediate_val)
#                diagnosis[sign_or_symptom + "s"][s] = val
            else:
                # for a negative distribution, we'll center (start/mean) the halfnorm -3SD (-3). No need
                #  to subract since it's going forward
                intermediate_val = scipy.stats.halfnorm(loc=-3*SD).rvs() # (to recenter at the top of the CDF)
                val = scipy.stats.norm().cdf(intermediate_val)
#                diagnosis[sign_or_symptom + "s"][s] = val
        # If it's boolean, pick whether the value will be correct or incorrect and assign it
        elif diagnosis[sign_or_symptom + "s"][s]['function'] == 'bool':
            # pick the correct value
            if baseline.rvs() < cutoff:
                if factors['inverse']:
                    val = 0
                else:
                    val = 1
            # pick the incorrect value
            else:
                if factors['inverse']:
                    val = 1
                else:
                    val = 0
#            diagnosis[sign_or_symptom + "s"][s] = val
        # for 3 levels, the bottom level is negative, the middle is 0 and the top is positive
        #  Because of this we assign based on SD
        elif diagnosis[sign_or_symptom + "s"][s]['function'] == 'step_3':
            # Pick a random value.
            intermediate_val = baseline.rvs()
            #  If the value is within 1SD, assign it the first level,
            if intermediate_val >= cutoff / float(3*SD): # since cutoff is 3SD, divide it by 3
                val = factors['levels'][0]
            #  If it's > 3SD, assign it the 3rd level
            elif intermediate_val > cutoff:
                val = factors['levels'][2]
            #  If it's between 1 and 3SD, assign it the 2nd level
            else:
                val = factors['levels'][1]
#            diagnosis[sign_or_symptom + "s"][s] = val
        elif diagnosis[sign_or_symptom + "s"][s]['function'] == 'step_10':
            # choose a value.
            intermediate_val = baseline.rvs()
            #Divide the space between the mean and 3SD evenly between 9 levels
            rng = (cutoff - mean) / float(9) # 9 is the number of samples less than 3SD
            lvl = int(intermediate_val/rng)
            # assign based on the bucket of the range
            # if it's over 3SD, assign the 10th level
            if lvl > 9:
                # previously, incorrect values always returned .1.  This is incorrect.  If the value is incorrect,
                #  It should be randomly chosen.  Otherwise .1 is disproportionately represented and likely to be
                #   'picked' as the correct value in model building
                val = np.random.choice(factors['levels'])
            else:
                try:
                    val = factors['levels'][lvl]
                except:
                    print rng, factors['levels'], lvl
                    raise
#            diagnosis[sign_or_symptom + "s"][s] = val
        else:
            raise KeyError("Function {0} not found in functions list.".format(diagnosis[sign_or_symptom + "s"][s]['function']))

        return s, val
#        return s, diagnosis[sign_or_symptom + "s"][s]

def main():
    logging.info('Beginning main loop.')
    # nothing happens here.  Use the class.
    logging.info('Ending main loop.')

if __name__ == "__main__":
    main()    
