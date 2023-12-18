import os
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.cegpy.trees import event, staged
from src.cegpy.graphs import ceg
import itertools
from itertools import combinations, chain, product
import networkx 
from networkx.algorithms.components.connected import connected_components

def show_event_tree(data):
    '''Plot the event tree for the data'''
    et = event.EventTree(data)
    return(et.create_figure())

def create_individual_hyperstages(hyperstage):
    '''Creates multible hyperstages one for each hyperset in the hyperstage'''
    new_hyperstage = {}
    for k in range(len(hyperstage)):
        temp_hyperstage = copy.deepcopy(hyperstage)
        del temp_hyperstage[k]
        new_hyperstage[k] = [[item] for sublist in temp_hyperstage for item in sublist]
        new_hyperstage[k].append(hyperstage[k])
    return new_hyperstage

# def run_multible_whac_for_each_hyperstage(st, K_maxs, prior_weight, new_hyperstage, hyperstage):
#     # initialize the set of stage trees
#     staged_trees={}
#     possible_mergings = {}
#     loglikelihoods = {}
#     # run code seperately for each hyperstage
#     for h in range(0, len(hyperstage)):
#         staged_trees[h] = {}
#         possible_mergings[h] = []
#         loglikelihoods[h] = []
#         # for each hyperstage run w-hac K_maxs[h] times 
#         for k in range(0, K_maxs[h]):
#             staged_trees[h][k] = copy.deepcopy(st)
#             # run w-hac
#             staged_trees[h][k].calculate_wHAC_transitions(hyperstage = new_hyperstage[h], alpha = prior_weight)
#             # sort the merged situations so that they can be compared
#             staged_trees[h][k].ahc_output['Merged Situations'] = sorted([sorted(list(x)) for x in staged_trees[h][k].ahc_output['Merged Situations']])
#             # check if staging has been achived before
#             if staged_trees[h][k].ahc_output['Merged Situations'] not in possible_mergings[h]:
#                 # if not add record the output
#                 possible_mergings[h].append(staged_trees[h][k].ahc_output['Merged Situations'])
#                 loglikelihoods[h].append(staged_trees[h][k].ahc_output["Loglikelihood"])
#         print(possible_mergings[h], loglikelihoods[h])
#     return possible_mergings, loglikelihoods


def run_multiple_whac_for_each_hyperstage(st, k_maxs, prior_weight, new_hyperstage, hyperstage):
    # initialize the overall results dictionaries
    staged_trees = {}
    possible_mergings = {}
    loglikelihoods = {}
    
    # iterate over each hyperstage
    for h, hyperstage_value in enumerate(hyperstage):
        staged_trees[h] = {}
        possible_mergings[h] = []
        loglikelihoods[h] = []

        # iterate k_maxs[h] times for the current hyperstage
        for k in range(k_maxs[h]):
            staged_trees[h][k] = copy.deepcopy(st)
            staged_trees[h][k].calculate_wHAC_transitions(hyperstage=new_hyperstage[h], alpha=prior_weight)
            staged_trees[h][k].ahc_output['Merged Situations'] = sorted([sorted(list(x)) for x in staged_trees[h][k].ahc_output['Merged Situations']])

            if staged_trees[h][k].ahc_output['Merged Situations'] not in possible_mergings[h]:
                possible_mergings[h].append(staged_trees[h][k].ahc_output['Merged Situations'])
                loglikelihoods[h].append(staged_trees[h][k].ahc_output["Loglikelihood"])

        print(possible_mergings[h], loglikelihoods[h])

    return possible_mergings, loglikelihoods

# def format_staging_output(st, possible_mergings, new_hyperstage, hyperstage):
#     # format stagings so that we can see potential stagings of each hyperstage
#     staging_output={}
#     # for each hyperstage
#     for h in range(0,len(hyperstage)):
#         staging_output[h]=[]
#         for staging in possible_mergings[h]:
#             temp_staging_output=[]
#             # if staging is in the hyperstage of interest include it if not dont
#             for stage in staging:
#                 if stage == hyperstage[h]:
#                     temp_staging_output.append(stage)
#                 elif stage not in new_hyperstage[h]:
#                     temp_staging_output.append(stage)
#             staging_output[h].append(temp_staging_output)
#     return staging_output

def format_staging_output(st, possible_mergings, new_hyperstage, hyperstage):
    staging_output = {}

    for h in range(len(hyperstage)):
        staging_output[h] = [
            # Filter and format stagings based on hyperstage and new_hyperstage
            [stage for stage in staging if stage == hyperstage[h] or stage not in new_hyperstage[h]]
            for staging in possible_mergings[h]
        ]

    return staging_output


def get_set_of_well_performing_stagings(staging_output ,loglikelihoods, alpha):
    # remove models which are non well performing
    well_performing_loglikelihoods = {}
    well_performing_staging = {}
    indexes = {}

    for h in range(0,len(loglikelihoods)):
        #Calculate the log Bayes factors for each log-likelihood in the hyperstage
        log_bayes_factors = [loglikelihood - loglikelihoods[h][0] for loglikelihood in loglikelihoods[h]]
        bayes_factors = [np.exp(loglikelihood) for loglikelihood in log_bayes_factors]

        # Select the Bayes factors that meet the well-performing criterion
        best_BFs = [bf for bf in bayes_factors if alpha * bf > max(bayes_factors)]

        # Check if any stagings are removed due to poor performance
        if len(best_BFs) != len(bayes_factors):
            print(len(bayes_factors)-len(best_BFs),' staging removed for hyperstage', h,'due to poor performance' )

        # Find the indexes of well-performing stagings
        indexes[h] = [i for i in range(0,len(bayes_factors)) if bayes_factors[i] in best_BFs]

        # Retrieve the well-performing stagings and log-likelihoods using the indexes
        well_performing_staging[h] = [staging_output[h][indexes[h][j]] for j in range(0,len(best_BFs))]
        well_performing_loglikelihoods[h] = [loglikelihoods[h][indexes[h][j]] for j in range(0,len(best_BFs))]
    return well_performing_staging, well_performing_loglikelihoods


def get_model_weights_from_loglikelihoods(loglikelihoods):
    # normalise loglikelihoods to give model weights
    model_weights = {}
    for h in range(0,len(loglikelihoods)):
        # If there is only one log-likelihood, assign a model weight of 1
        if len(loglikelihoods[h]) == 1:
            model_weights[h] = np.array([[1]])
        else :
            # Calculate the log Bayes factors and Bayes factors
            log_bayes_factors = [loglikelihood - loglikelihoods[h][0] for loglikelihood in loglikelihoods[h]]
            bayes_factors = [np.exp(loglikelihood) for loglikelihood in log_bayes_factors]
            # Normalize the model weights by dividing each weight by the sum of all weights
            model_weights[h] = bayes_factors/sum(bayes_factors)
    return model_weights



def Bayesian_model_averaging_CEGs(st, prior_weight=[], K_max=100, alpha=20, hyperstage=[]):
    # Set default values if not provided
    if hyperstage == []:
        hyperstage = st._create_default_hyperstage()
    K_maxs = [K_max * len(hyperset) for hyperset in hyperstage]
    if prior_weight == []:
        prior_weight = max(st.categories_per_variable.values())
    
    # Create multiple hyperstages, one for each hyperset in the hyperstage
    new_hyperstage = create_individual_hyperstages(hyperstage)
    
    # Run w-hac for each hyperset
    possible_mergings, loglikelihoods = run_multiple_whac_for_each_hyperstage(st, K_maxs, prior_weight, new_hyperstage, hyperstage)
    
    # Format staging output
    staging_output = format_staging_output(st, possible_mergings, new_hyperstage, hyperstage)
    
    # Get well-performing stagings based on loglikelihoods
    staging_output, loglikelihoods = get_set_of_well_performing_stagings(staging_output, loglikelihoods, alpha)
    
    # Calculate model weights from loglikelihoods
    model_weights = get_model_weights_from_loglikelihoods(loglikelihoods)
    
    return staging_output, model_weights, loglikelihoods


def get_full_model_weights(model_weights, staging_output):
    # Prepare the staging outputs
    list_staging_outputs = [staging_output[h] for h in range(0, len(staging_output))]
    temp_full_staging_output = [element for element in product(*list_staging_outputs)]
    full_staging_output = {}

    # Format the full staging output
    for h in range(0, len(temp_full_staging_output)):
        full_staging_output[h] = [item for sublist in temp_full_staging_output[h] for item in sublist]

    model_weights_list = {key: value.tolist() for key, value in model_weights.items()}
    list_arrays = [model_weights[h] for h in model_weights]
    all_combinations = list(product(*list_arrays))

    full_model_weights = [np.prod(combination) for combination in all_combinations]
    # list_model_weights = [list([model_weights_list[h]]) for h in range(0, len(model_weights_list))]
    # temp_list_model_weights = [element for element in product(*list_model_weights)]
    # weight_full_staging_output = {}

    # Format the full staging output
    # for h in range(0, len(temp_list_model_weights)):
    #     weight_full_staging_output[h] = [item for sublist in temp_list_model_weights[h] for item in sublist]


    # Calculate the full model weights
    # full_model_weights_prod = product(*list_model_weights)
    # full_model_weights = np.prod(list(full_model_weights_prod), axis=1)
    # full_model_weights = full_model_weights[0].flatten()

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
