from math import comb
from src.cegpy.trees import event, staged
from src.cegpy.graphs import ceg
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from itertools import combinations, chain, product
import networkx 
from networkx.algorithms.components.connected import connected_components

def show_event_tree(data):
    '''Plot the event tree for the data'''
    et = event.EventTree(data)
    return(et.create_figure())

def create_indivual_hyperstages(st, hyperstage):
    new_hyperstage = {}
    for k in range(0,len(hyperstage)):
        temp_hyperstage = copy.deepcopy(hyperstage)
        del temp_hyperstage[k]
        new_hyperstage[k] = [[item] for sublist in temp_hyperstage for item in sublist]
        new_hyperstage[k].append(hyperstage[k])
    return new_hyperstage

def run_multible_whac_for_each_hyperstage(st, K_maxs, prior_weight, new_hyperstage, hyperstage):
    # initialize the set of stage trees
    staged_trees={}
    possible_mergings = {}
    loglikelihoods = {}
    # run code seperately for each hyperstage
    for h in range(0, len(hyperstage)):
        staged_trees[h] = {}
        possible_mergings[h] = []
        loglikelihoods[h] = []
        # for each hyperstage run w-hac K_maxs[h] times 
        for k in range(0, K_maxs[h]):
            staged_trees[h][k] = copy.deepcopy(st)
            # run w-hac
            staged_trees[h][k].calculate_wHAC_transitions(hyperstage = new_hyperstage[h], alpha = prior_weight)
            # sort the merged situations so that they can be compared
            staged_trees[h][k].ahc_output['Merged Situations'] = sorted([sorted(list(x)) for x in staged_trees[h][k].ahc_output['Merged Situations']])
            # check if staging has been achived before
            if staged_trees[h][k].ahc_output['Merged Situations'] not in possible_mergings[h]:
                # if not add record the output
                possible_mergings[h].append(staged_trees[h][k].ahc_output['Merged Situations'])
                loglikelihoods[h].append(staged_trees[h][k].ahc_output["Loglikelihood"])
    return possible_mergings, loglikelihoods


def format_staging_output(st, possible_mergings, new_hyperstage, hyperstage):
    # format stagings so that we can see potential stagings of each hyperstage
    staging_output={}
    # for each hyperstage
    for h in range(0,len(hyperstage)):
        staging_output[h]=[]
        for staging in possible_mergings[h]:
            temp_staging_output=[]
            # if staging is in the hyperstage of interest include it if not dont
            for stage in staging:
                if stage == hyperstage[h]:
                    temp_staging_output.append(stage)
                elif stage not in new_hyperstage[h]:
                    temp_staging_output.append(stage)
            staging_output[h].append(temp_staging_output)
    return staging_output

def get_set_of_well_performing_stagings(staging_output ,loglikelihoods, alpha):
    # remove models which are non well performing
    well_performing_loglikelihoods = {}
    well_performing_staging = {}
    indexes = {}
    for h in range(0,len(loglikelihoods)):
        log_bayes_factors = [loglikelihood - loglikelihoods[h][0] for loglikelihood in loglikelihoods[h]]
        bayes_factors = [np.exp(loglikelihood) for loglikelihood in log_bayes_factors]
        best_BFs = [bf for bf in bayes_factors if alpha * bf > max(bayes_factors)]
        if len(best_BFs) != len(bayes_factors):
            print(len(bayes_factors)-len(best_BFs),' staging removed for hyperstage', h,'due to poor performance' )
        indexes[h] = [i for i in range(0,len(bayes_factors)) if bayes_factors[i] in best_BFs]
        well_performing_staging[h] = [staging_output[h][indexes[h][j]] for j in range(0,len(best_BFs))]
        well_performing_loglikelihoods[h] = [loglikelihoods[h][indexes[h][j]] for j in range(0,len(best_BFs))]
    return well_performing_staging, well_performing_loglikelihoods

def get_model_weights_from_loglikelihoods(loglikelihoods):
    # normalise loglikelihoods to give model weights
    model_weights = {}
    for h in range(0,len(loglikelihoods)):
        if len(loglikelihoods[h]) == 1:
            model_weights[h] = 1
        else :
            log_bayes_factors = [loglikelihood - loglikelihoods[h][0] for loglikelihood in loglikelihoods[h]]
            bayes_factors = [np.exp(loglikelihood) for loglikelihood in log_bayes_factors]
            model_weights[h] = bayes_factors/sum(bayes_factors)
    return model_weights

def Bayesian_model_averaging_CEGs(st, prior_weight = [], K_max = [], alpha = 20, hyperstage =[]):
    if hyperstage == []:
        hyperstage = st._create_default_hyperstage()
    if K_max == []:
        K_max = 100
    K_maxs=[K_max*len(hyperset) for hyperset in hyperstage]
    if prior_weight == []:
        prior_weight = max(st.categories_per_variable.values())
    new_hyperstage = create_indivual_hyperstages(st, hyperstage)
    possible_mergings, loglikelihoods = run_multible_whac_for_each_hyperstage(st, K_maxs, prior_weight, new_hyperstage, hyperstage)
    staging_output = format_staging_output(st, possible_mergings, new_hyperstage, hyperstage)
    staging_output, loglikelihoods = get_set_of_well_performing_stagings(staging_output, loglikelihoods, alpha)
    model_weights = get_model_weights_from_loglikelihoods(loglikelihoods)
    return staging_output, model_weights, loglikelihoods

def get_full_model_weights(st, model_weights, staging_output):
    list_staging_outputs=[staging_output[h] for h in range(0,len(staging_output))]
    temp_full_staging_output=[element for element in product(*list_staging_outputs)]
    full_staging_output={}
    for h in range(0,len(temp_full_staging_output)):
        full_staging_output[h] = [item for sublist in temp_full_staging_output[h] for item in sublist]
    list_model_weights=[[model_weights[h]] for h in range(0,len(model_weights))]

    full_model_weights_prod = product(*list_model_weights)
    full_model_weights = np.product(list(full_model_weights_prod), axis = 1)
    full_model_weights = full_model_weights[0].flatten()
    return full_staging_output, full_model_weights

def get_staging_intersection(staging_output):
    intersection = {}
    for h in range(0,len(staging_output)):
        if len(staging_output[h]) == 1:
            intersection[h] = staging_output[h][0]
        else:
            situations = [item for sublist in staging_output[h][0] for item in sublist]
            found_situations = []
            situations_colour = []
            for s in range(0,len(situations)):
                if situations[s] not in found_situations:
                    for c in staging_output[h][0]:
                        if situations[s] in c:
                            c.sort()
                            situation_colour = c
                    for m in range(1,len(staging_output[h])):
                        for c in staging_output[h][m]:
                            if situations[s] in c:
                                c.sort()
                                situation_colour = [x for x in situation_colour if x in c]
                    found_situations += [sit for sit in situation_colour]
                    situations_colour.append(situation_colour)
            intersection[h] = situations_colour

    return intersection

def get_staging_union(staging_output):
    union = {}
    for h in range(0,len(staging_output)):
        if len(staging_output[h]) == 1:
            union[h] = staging_output[h][0]
        else:
            situations = [item for sublist in staging_output[h][0] for item in sublist]
            situations_colour = []
            for s in range(0,len(situations)):
                for c in staging_output[h][0]:
                    if situations[s] in c:
                        c.sort()
                        situation_colour = c
                for m in range(1,len(staging_output[h])):
                    for c in staging_output[h][m]:
                        if situations[s] in c:
                            c.sort()
                            situation_colour = list(set(situation_colour).union(set(c)))
                situations_colour.append(situation_colour)
            G = networkx.Graph()
            for cluster in situations_colour:
                G.add_nodes_from(cluster)
                G.add_edges_from(zip(cluster[:-1], cluster[1:]))
            merged_situations_colour = list(connected_components(G))
            merged_situations_colour = [list(x) for x in merged_situations_colour]
            
            union[h] = merged_situations_colour

    return union
