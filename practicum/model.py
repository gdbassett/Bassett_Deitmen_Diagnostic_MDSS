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


    def __init__(self):
        pass


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
                g1.add_edge(sign, record['diagnosis'], value=record['signs'][sign])
                signs[sign] += 1
                if type(record['signs'][sign]) not in [int, float, np.float64]:
                    raise TypeError('Sign/Symptom {0} must be some type of number rather than {1} for {2}'.format(
                        sign, type(record['signs'][sign]), record))
            for symptom in record['symptoms']:
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





'''
    model = None  # in the form of a graph

#    def new_model(self):
#        """
#        :return: Nothing
#
#        Takes nothing, returns an empty model for training.
#        """
#        pass

    def load_model(self, model = NEODB):
        """

        :param model: Neo4j database URI as string
        :return: booling success

        Takes a string refering to to a neo4j database and save the database handle to the class
        """
        try:
#            self.model = py2neoGraph(model)
            logging.info("Model Loaded.")
            return True
        except Exception as e:
            logging.error(e.__str__)
            logging.info("Model failed to load.")
            return False


    def sign_symptom_training(self, sign_symptom_map):
        """

        :param sign_symptom_map: dictionary of keys of diagnosis and list of tuples of (signs/symptoms, confidence, value)  (value is only used for continuous variables)
        :return: booling success

        Takes a single mapping of a sign/symptom to a diagnosis adds it to the Model
        """
        pass  # TODO
        #  TODO : For continuous variables, the total for that variable (regardless of relation to the diagnosis), may need to be kept to normalize Y axis.

        # TODO: (Need a way to categorize signs/symptoms as continuous, factors, or bool)

        # TODO: if needed, create the sign/symptom & label it's type (sign or symptom), it's class (factor, continuous, or bool)

        # TODO: If needed, create a diagnosis

        # TODO: if it doesn't exist, create an edge between the sign/symptom and the diagnosis.  Create initial confidence (factor table, function, or bool)
        # TODO:  If you cannot incrimentally update, the model will need to keep the entire value set and at the end build a distribution for each of the continuous variables

        # TODO: if it already exists update the confidence (factor table, function, or bool count)

        # NOTE: sign/symptom - diagnosis relationships may be negative in confidence indicating a lacking.  However negative and positive relationships should have their own edges.


    def test_symptom_training(self, test_symptom_map):
        """

        :param test_symptom_map: dictionary of keys of tests and list of tuples of (symptoms, confidence)
        :return: booling success

        Takes a single mapping of a sign/symptom to a diagnosis adds it to the Model
        """
        pass  # TODO


    def diagnosis_treatment_training(self, diagnosis_treatment_map):
        """

        :param diagnosis_treatment_map: dictionary of keys of treatment and values of (diagnosis, impact)
        :return: booling success

        Takes a single mapping of a treatment to diganosis and adds it to the Model
        """
        pass  # TODO


    def query(self, signs_symptoms):
        """

        : param signs_symptoms: A list of signs and/or symptoms
        : return: a dictionary of form {diagnoses: {diagnosis: score}, tests:{test: score}, treatments:{treatment:score}}
        """

        d = {}
        d['diagnosis'] = query_diagnosis(signs_symptoms)
        d['test'] = query_test(d['diagnosis'])
        d['treatements'] = query_treatments(d['diagnosis'])

        return d


    def query_diagnosis(self, signs_symptoms):
        """

        : param signs_symptoms: A list of signs and/or symptoms
        : return: a dictionary of form {diagnosis: score}
        """
        pass  # TODO
        # TODO: Get all signs & symptoms from graph

        # TODO: Get all out edges associated with each sign/symptom

        # TOOD: If necessary, calculate distribution

        # TODO: Normalize to out degree (bool) or feature/distribution (categorical/conditnuous) # May be done during import, but only 

        # TODO: Find the single value of categorical/continuous signs/symptoms as confidence

        # TOOD: sum the confidence by destination of each edge

        # TODO: find all edges into the diagnosis.  Reduce the confidence summation of each diagnosis by the sum of the confidences of it's missing edges.

        # TODO: Return the diagnosis:score dictionary

    def query_test(self, diagnoses):
        """

        : param diagnoses: a dictionary of diagnoses and scores associated with their likelihood
        : return: a dictionary of form {test: score}
        """
        pass  # TODO


    def query_treatments(self, diagnoses):
        """

        : param diagnoses: a dictionary of diagnoses and scores associated with their likelihood
        : return: a dictionary of form {treatment:score}
        """
        pass  # TODO


    def validate_model(sign_symptoms, actual):
        """

        : param sign_symptoms: a list of lists of signs/symptoms
        : param actual: a list of actual diagnosis
        : return: a list of bools representing whether the actual diagnosis was in the top 5 diagnoses
        """
        pass  # TODO

        diagnosis = self.query_diagnosis(signs_symptoms)

        # TODO: Sort the diagnosis by score

        # TODO: If the diagnosis is the top diagnosis, incriment the top1 counter
        
        # TODO: if the diagnosis is in the top 5, increment the top5 counter
        
        # TODO: Increment the total counter

        # TODO: Return the 1% and 5% precisions.  (Could we also calculate recall?)
'''

def main():
    logging.info('Beginning main loop.')

    logging.info('Main loop complete.')
if __name__ == "__main__":
    main()    
