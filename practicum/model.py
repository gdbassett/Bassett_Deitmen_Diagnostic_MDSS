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

 CAVEAT:
 - The model assumes that signs and symptoms are known to be continuous or categorical and one-sided (bool & log), 2-sided (normal & 3-level), or progressive (linear & 10-step)

"""
# PRE-USER SETUP
import numpy as np

########### NOT USER EDITABLE ABOVE THIS POINT #################


# USER VARIABLES
NEODB = "http://192.168.121.134:7474/db/data"

# SET RANDOM SEED 
np.random.seed(5052015)

## TRUTH DATA STATIC VARIABLES, based on physician recommendation
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
#SYMPTOM_VS_SIGN_RELATIVE_WEIGHT = .7 # symptom_weight/sign_weight i.e. we trust signs 100% and symptoms 70%
SYMPTOM_VS_SIGN_RELATIVE_WEIGHT = .6/.85 # symptom_weight/sign_weight i.e. we trust signs 100% and symptoms 70%

## RECORDS STATIC VARIABLES


########### NOT USER EDITABLE BELOW THIS POINT #################


## IMPORTS
from py2neo import Graph as py2neoGraph
import networkx as nx
import argparse
import logging
from flask import Flask
from flask.ext.restful import reqparse, Resource, Api, abort
from collections import defaultdict, Counter
import copy
import scipy.stats
from scipy.optimize import curve_fit

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
class decision_support_system():
    model = None
    records_graph = None
    signs = None
    symptoms = None
    diagnoses = None
    test_model = None


    def __init__(self):
        pass


    def add_treatments_to_model_nx(self, treatment_data):
        # If the model hasn't been built, error
        if self.model is None:
            raise TypeError("Model does not appear to be built.  Please build the model before incorporating treatment"
                            " data.")

        for treatment in treatment_data.keys():
            if treatment not in self.model.nodes():
                self.model.add_node(treatment, {'type':'treatment'})
            for dss in treatment_data[treatment].keys():  # dss = diagnosis, sign, symptom
                if dss not in self.model.nodes():
                    self.model.add_node(dss, {'type':treatment_data[treatment][dss]['type']})
                self.model.add_edge(treatment, dss, attr_dict={'impact':treatment_data[treatment][dss]['impact']})


    def add_tests_to_model_nx(self, test_data):
        # If the model hasn't been built, error
        if self.model is None:
            raise TypeError("Model does not appear to be built.  Please build the model before incorporating test data.")

        # For each test, get the diagnoses it's linked to through signs
        for test in test_data.keys():
            if test not in self.model.nodes():
                self.model.add_node(test, {'type':'test'})
            for sign in test_data[test].keys():
                    if sign not in self.model.nodes():
                        self.model.add_node(sign, {'type':'sign'})
                    self.model.add_edge(test, sign, attr_dict={'confidence':test_data[test][sign]['confidence']})


    def build_model_nx(self, g1=None):
        """
        :param g1: a networkx MultiDiGraph with nodes for signs/symptoms and diagnoses and edges from signs/symptoms to the diagnoses
        :return: a networkx DiGraph representing the model
        """

            # With G2, we take each sign/symptom relationship and compress multi-edges down to a single edge with a
        #  probability density distribution function for the edge.  This will be what we use for classifying in the
        #  model.

        # Graph 2 for condensed relationships and functional probabilities
        g2 = nx.DiGraph()

        # in case records_graph was not filled in, use the one from the object
        if g1 is None:
            g1 = self.records_graph

        # Store the graph to an internal variable.  This could be used for
        relationships = set(g1.edges())

        # this is a bit of a hack with a 4 relationship filter set arbitrarily
        remove_rels = set()
        for relationship in relationships:
            if len(g1.edge[relationship[0]][relationship[1]]) <= 4:  # 4 is arbitrary @ 1 million records, tru are usually > 20 or 30 & false are normally < 5
                remove_rels.add(relationship)
        relationships = relationships.difference(remove_rels)


        # for each edge pair
        for relationship in relationships:
            edges = copy.deepcopy(g1.edge[relationship[0]][relationship[1]])
            # build a distribution (histogram) of the values
            values = Counter([edges[x]['value'] for x in edges])
            x = np.asarray(values.keys())
            y = np.asarray(values.values())
            y_norm = y/float(sum(y))

            # Tried using curve_fit (immediately below), but requires binned data so chose norm.fit()
            #params, pcov = curve_fit(lambda x_data, a, b: scipy.stats.halfnorm.pdf(x_data, loc=a, scale=b), x_in, [n/float(sum(y_in)) for n in y_in])
            # create distribution
            mu, std = scipy.stats.norm.fit([edges[n]['value'] for n in edges])


            # create the normal fit
            # if there is data beyond mu +- std, keep it
            if len([n for n in x if n < mu - std]) > 0 and len([n for n in x if n > mu + std]) > 0:
                normalize = scipy.stats.norm(loc=mu, scale=std).pdf(mu)  # normalize to the mean
                f = lambda x_data : scipy.stats.norm(loc=mu, scale=std).pdf(x_data) / float(normalize)
            # else, scrap it and create a halfnorm (or just use the cdf
            else:
                # Can't get a good plot out of halfnorm so using full norm & normalizing the to the max in the data
                normalize = scipy.stats.norm(loc=mu, scale=std).cdf(max(x))  # normalze to the maximum of the data
                f = lambda x_data: scipy.stats.norm(loc=mu, scale=std).cdf(x_data) / float(normalize)


            # store density function on edge
            g2.add_edge(relationship[0], relationship[1], value = f, relationship_count = len(edges))

            #f when called with a value for the sign/symptom will return a confidence score that should be normalized to 1

        # add a type to each node to make coloring/filtering them easier
        # TODO: add edge w/ type=sign, type=symptom, type=diagnosis, etc
        # for node in g2.nodes():
        #    try:
        #       g2.node[node]['type'] = g1.node[node]['type']
        #    except KeyError:  # or whatever a missing node error from a graph is
        #       g2.node[node]['type'] = node.split("_",1)[0]
        #TODO: remove this and instead carry through the 'type' from previous the injestion graph
        for node in g2.nodes():
            g2.node[node]['type'] = node.split("_",1)[0]

        # for each node, count the number of relationships it had from g1
        for node in g2.nodes():
            type = g2.node[node]['type']
            if type in ['sign', 'symptom']:
                g2.node[node]['relationship_count'] = g1.out_degree(node)
            elif type in ['diagnosis']:
                g2.node[node]['relationship_count'] = g2.in_degree(node)


        # boom.  model.
        return g2


    def injest_records_nx(self, records):
        """
        :param records: Takes a list of medical records (synthetic or otherwise) in the form {diagnosis:"", signs:{}, symptoms:{}}
        :return: a networkx graph representing the records
        """
        # Graph 1 for all relationships
        g1 = nx.MultiDiGraph()
        # Define dictionaries to store the signs, symptoms, and diagnoses and cardinality
        signs = defaultdict(int)
        symptoms = defaultdict(int)
        diagnoses = defaultdict(int)



        # for each record, connect signs/symtoms to their respective diagnoses with value
        for record in records:
            diagnoses['diagnosis'] += 1
            for sign in record['signs']:
                # TODO: add edge w/ type=sign, type=symptom, type=diagnosis, etc
                # g1.add_node(sign, type='sign')
                # g1.add_node(record['diagnosis'], type='diagnosis')
                g1.add_edge(sign, record['diagnosis'], value=record['signs'][sign])
                signs[sign] += 1
                if type(record['signs'][sign]) not in [int, float, np.float64]:
                    raise TypeError('Sign/Symptom {0} must be some type of number rather than {1} for {2}'.format(
                        sign, type(record['signs'][sign]), record))
            for symptom in record['symptoms']:
                # TODO: add edge w/ type=sign, type=symptom, type=diagnosis, etc
                # g1.add_node(symptom, type='symptom')
                # g1.add_node(record['diagnosis'], type='diagnosis')
                g1.add_edge(symptom, record['diagnosis'], value=record['symptoms'][symptom])
                symptoms[symptom] +=1

        # Actual tallies can be found by querying the degree of self.record_graph
        self.signs = signs.keys()
        self.symptoms = symptoms.keys()
        self.diagnoses = diagnoses.keys()
        self.records_graph = g1

        # g1 now has a bipartite multi directed graph with edges from signs or symptoms to diagnosis
        #  bearing the value in a single record
        return g1


    def query_nx_model(self, record):

        # get potential diagnoses
        potential_diagnoses = set()
        for sign in record['signs']:
            if self.model.has_node(sign):
                potential_diagnoses = potential_diagnoses.union(self.model.successors(sign))
        for symptom in record['symptoms']:
            if self.model.has_node(symptom):
                potential_diagnoses = potential_diagnoses.union(self.model.successors(symptom))

        # filter the diagnoses by primary symptom
        pass  # TODO

        # score the diagnoses
        scores = defaultdict(float)
        for diagnosis in potential_diagnoses:
            sign_edge_cnt = 0
            symptom_edge_cnt = 0
            # Below replaced with total count on diagnosis nodes
            '''
            for predecessor in self.model.predecessors(diagnosis):
                if 'sign' in predecessor:
                    sign_edge_cnt += self.model.edge[predecessor][diagnosis]['relationship_count']
                elif 'symptom' in predecessor:
                    symptom_edge_cnt += self.model.edge[predecessor][diagnosis]['relationship_count']
            '''
            sign_score = 0
            for sign in record['signs']:
                if self.model.has_edge(sign, diagnosis):
                    # edge values form signs/symptoms
                    sign_score += self.model.edge[sign][diagnosis]['value'](record['signs'][sign])  # call the relationship model
                    # weighted by #edges/ sum(#edges of diagnosis)
                    # Below replaced with total count on diagnosis nodes
                    '''
                    score = score * self.model.edge[sign, diagnosis]['relationship_count'] / float(sign_edge_cnt)
                    '''
                    sign_score *= self.model.edge[sign][diagnosis]['relationship_count'] / \
                            float(self.model.node[diagnosis]['relationship_count'])
                    # weighted for relative importance of particular sign (i.e. chest pain > finger pain)
                    sign_score *= self.model.node[sign].get('relative_weight', 1)
                    # weighted less for symptoms
                    sign_score *= 1  # SYMPTOM_VS_SIGN_RELATIVE_WEIGHT value below is normalized to this one (1)
            symptom_score = 0
            for symptom in record['symptoms']:
                if self.model.has_edge(symptom, diagnosis):
                    # edge values form signs/symptoms
                    symptom_score += self.model.edge[symptom][diagnosis]['value'](record['symptoms'][symptom])  # call the relationship model
                    # weighted less for symptoms
                    # Below replaced with total count on diagnosis nodes
                    '''
                    score = score * self.model.edge[symptom, diagnosis]['relationship_count'] / float(symptom_edge_cnt)
                    '''
                    try:
                        symptom_score *= self.model.edge[symptom][diagnosis]['relationship_count'] / \
                                float(self.model.node[diagnosis]['relationship_count'])
                    except:
                        print diagnosis, symptom
                        raise
                    # weighted for relative importance of particular sign (i.e. chest pain > finger pain)
                    symptom_score *= self.model.node[symptom].get('relative_weight', 1)
                    # weighted by #edges/ sum(#edges of diagnosis)
                    symptom_score *= SYMPTOM_VS_SIGN_RELATIVE_WEIGHT

            scores[diagnosis] = sign_score + symptom_score

        return scores


    def query_tests(self, record, potential_diagnoses, diagnoses_to_consider=30, differential_type='individual'):
        """

        :param record: The dictionary record for which tests are identified
        :param potential_diagnoses: The ouptut of the query_nx_model function
        :param diagnoses_to_consider: The number of diagnoses to consider relevant
        :param differential_type: How to determine tests.  'individual' indicates a set of tests to uniquely test each diagnosis.  'split' produces a test which best splits the diagnoses to consider, 'subset' (NOT YET IMPLEMENTED) will produce a minimal subset of tests to uniquely identify a diagnosis among the diagnoses to consider.
        :return: list of tuples of (diagnosis, list of tests in priority order)
        """

        if differential_type not in ('individual', 'subset', 'split'):
            raise ValueError("differential_type must be 'individual' meaning attempt to find tests which attempt to"
                             "definitively confirm or deny single diagnoses or 'subset' meaning tests meant to "
                             "subset the top potential diagnoses, or 'split' for a single test which best splits"
                             "the diagnoses.")


        # Subset to the potential diagnoses to be subsetted
        L = sorted(potential_diagnoses.items(), key=lambda (k, v): v)
        L =map(lambda (k,v): k, L)
        L = L[0:diagnoses_to_consider]
        # get signs
        potential_tests = set()
        test_model = nx.DiGraph()
        for diagnosis in L:
            for sign in self.model.predecessors(diagnosis):
                # remove the symptoms
                # remove signs in the record
                if self.model.node[sign]['type'] == 'sign' and sign not in record['signs'].keys():
                    for test in self.model.predecessors(sign):
                        if self.model.node[test]['type'] == 'test':
                            confidence = self.model.edge[test][sign]['confidence']
                            # TODO: Could potentially weight the confidence with the sign->diagnosis function here.
                            #         One option is to weight by how much the function differs from the uniform distribution
                            #         This would be how well the function can differentiate.
                            test_model.add_edge(test, diagnosis, attr_dict={'confidence':confidence})
                            potential_tests.add(test)

        if differential_type in ('individual'):
            ind_tests = list()
            # starting with the top diagnosis and working our way down
            for i in range(len(L)):
                diagnosis = []
                # get all of the tests for that diagnosis.
                diag_tests = test_model.predecessors(L[i])
                # if zero, pass the logic and add the empty list
                if len(diag_tests) >= 0:
                    # Get edge degrees
                    diag_tests = [(t, test_model.out_degree(t)) for t in diag_tests]
                    # Get the tests that describe the minimum number of diagnoses
                    min_diag = min([x[1] for x in diag_tests])
                    diag_tests = [(t, test_model.edge[test][L[i]]['confidence']) for t, deg in diag_tests if deg <= min_diag]
                    # sort the list by confidence
                    diag_tests.sort(key=lambda tup: tup[1], reverse=True)
                    # put the list with the diagnosis in the overall list
                    diagnosis = ind_tests
                ind_tests.append((L[i], diagnosis))

                return ind_tests

        if differential_type in ('subset'):
            # TODO: This script finds  a set of tests which completely unique
            # could build an array with tests as columns and diagnoses as rows and find the minimum columsn to produce a candidate key
            raise AttributeError("subset function not yet implemented.")


        if differential_type in ('split'):
            split_test = (None, 100)  # 100 should ensure it is replaced
            target = diagnoses_to_consider/float(2)
            for t in potential_tests:
                diff = abs(target - test_model.out_degree(t))
                # if diff is closer to the bisection, keep that test
                if diff < split_test[1]:
                    split_test = (t, diff)
                # could do an elif if the distance is equal and try and find the test related to the highest diagnosis
                #  too much work for now though

            # get the diagnoses and scores for the test
            ret = []
            diags = test_model.successors(split_test[0])
            for i in range(len(L)):
                if L[i] in diags:
                    ret.append((L[i], [split_test[0]]))
                else:
                    ret.append((L[i], []))

            return ret

        raise StandardError("Reached the end of the function without returning  tests.")



    # TODO: create subgraph of sign/symptom/diagnosis <- treatment and iterate over it
    def query_treatments(self, potential_diagnoses, diagnoses_to_consider=20):
        # given a scored list of potential diagnosis,
        potential_treatments = list()  # list may need to be changed
        # for the top (diagnoses_to_consider) diagnoses:
        #   get successor treatments (record it as a diagnosis treatment)
        # get successor treatments to signs (record it as a sign/symptom treatment)
        #    filter by positive relationships (may want to keep negative relationships to show)
        # get successor treatments to symptoms (record it as a sign/symptom treatment)
        #   filter by positive relationships (may want to keep negative relationships to show)
        # for each potential_treatment:
        #   get all predecessors with negative relationships
        #   if the negative relationship is in the top (SOME CUTOFF.  SAY 99.7%): drop it
        #   else: record the location in the scored potential diagnoses list

        # add impact score

        # Order diagnoses by:
        # 1. It treats diagnoses
        # 1.1 It treats the most (diagnoses to consider) diagnoses
        # 1.2 Lowest negative effect location
        # 2. It treats signs/symptoms
        # 2.1 Sorted by negative effect location


        # Return potential treatments, ordered, with (sign/sympotm vs diagnosis, # treated, highest negative effect loc)

        pass


    def train_nx_model(self, records):
        """
        :param records: Takes a list of medical records (synthetic or otherwise) in the form {diagnosis:"", signs:{}, symptoms:{}}
        :return: a networkx graph representing the model
        """

        # From the list of records, build the records graph
        self.records_graph = self.injest_records_nx(records)
        g1 = self.records_graph

        # From the records graph, build the graph model
        self.model = self.build_model_nx(g1)
        return self.model



def main():
    logging.info('Beginning main loop.')

    logging.info('Main loop complete.')
if __name__ == "__main__":
    main()    
