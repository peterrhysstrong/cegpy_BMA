# %%
from distutils.log import error
from random import seed
from Bayesian_model_averaging_CEG import *
import pandas as pd
from src.cegpy.trees import event, staged
from src.cegpy.graphs import ceg
import copy
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial import distance
import networkx as nx
#%%
def make_binary(df):
    df_bin = pd.DataFrame()
    for col in df.columns:
        # df_bin_temp = pd.DataFrame()
        uni = df[col].unique()
        # uni = df[col].unique()[::-1]
        if len(uni) == 2:
            df_bin[str(col) + "_" + str(uni[0])] = df[col].apply(lambda x: uni[0] if x == uni[0] else uni[1])
        else:
            for i in range(0,len(uni)-1):
                if i == 0:
                    df_bin[str(col) + "_" + str(uni[i])] = df[col].apply(lambda x: uni[i] if x == uni[i] else "Other")
                elif i == len(uni)-2:
                    df_bin[str(col) + "_" + str(uni[i])] = df[col].apply(lambda x: uni[i] if x == uni[i] else uni[i+1])
                    df_bin[str(col) + "_" + str(uni[i])][df_bin[str(col) + "_" + str(uni[i-1])]!="Other"] = np.nan
                else:
                    df_bin[str(col) + "_" + str(uni[i])] = df[col].apply(lambda x: uni[i] if x == uni[i] else "Other")
                    df_bin[str(col) + "_" + str(uni[i])][df_bin[str(col) + "_" + str(uni[i-1])]!="Other"] = np.nan
    return df_bin
#%%
def get_missing_paths(df,bin_search=[]):
    '''trys to find missing paths although doesnt work in all cases and needs testing'''
    missing_paths = []
    uniques = [list(set(df[k])) for k in df.columns]
    all_paths=list(product(*uniques))        
    paths=df.drop_duplicates().values.tolist()
    all_paths_list = [list(ele) for ele in all_paths]
    Length_of_paths=len(all_paths_list[0])
    if bin_search!=[]:
        bin_start = [uniques.index(u) for u in uniques if "Other" in u and len(u)==2]
        bin_end = [uniques.index(u) for u in uniques if "Other" not in u and len(u)==3]
        for i in range(0,len(bin_start)):
            bin_section = [path[bin_start[i]:bin_end[i]+1] for path in all_paths_list]
            paths_in_bin_section = [[outcome] for outcome in uniques[bin_end[i]] if outcome == outcome] 
            for j in range(1,bin_end[i]-bin_start[i]+1):
                paths_in_bin_section = [["Other"] + path for path in paths_in_bin_section]
                other_outcome = [event for event in uniques[bin_end[i]-j] if event == event and event!="Other"] 
                paths_in_bin_section = paths_in_bin_section + [other_outcome + [float('nan')] * (len(paths_in_bin_section[0])-1)]
            allowed_paths = [path for path in range(0,len(bin_section)) if bin_section[path] in paths_in_bin_section]
            all_paths_list = np.array(all_paths_list)[allowed_paths].tolist()
    for i in range(0,Length_of_paths):
        temp_paths = [path[:Length_of_paths-i] for path in paths]
        new_missing_paths_list=[path[:Length_of_paths-i] for path in all_paths_list if path[:Length_of_paths-i] not in temp_paths]
        if new_missing_paths_list != []:
            missing_paths=[*new_missing_paths_list,*missing_paths]
    missing_paths = list(tuple(x) for x in missing_paths)
    missing_paths = sorted(missing_paths, key=len)
    return(missing_paths)

def sort_bin_paths(missing_paths,df):
    #find unique values
    uniques = [list(set(df[k])) for k in df.columns]
    #find there indexes with 3 outcomes not being other ie nan and two values
    bin_start = [uniques.index(u) for u in uniques if "Other" in u and len(u)==2]
    bin_end = [uniques.index(u) for u in uniques if "Other" not in u and len(u)==3]
    removed_paths = []
    for path in missing_paths:
        in_bin_section = 0
        for l in range(0,len(path)):
            if l in bin_start:
                must_nans = 0
                no_nans = 0
                in_bin_section = 1
            if in_bin_section == 1:
                if path[l] == np.nan:
                    if no_nans == 1:
                        removed_paths.append(missing_paths.index(path))
                        continue
                    must_nans = 1
                elif path[l] == "Other":
                    if must_nans ==1:
                        removed_paths.append(missing_paths.index(path))
                        continue
                    no_nans = 1 
                else:
                    if must_nans ==1:
                        removed_paths.append(missing_paths.index(path))
                        continue
                    must_nans = 1
    updated_missing_paths = [path for path in missing_paths if missing_paths.index(path) not in removed_paths]
    return(updated_missing_paths)
#%%

def get_priors(staged_tree,alpha=1):
    paths=nx.all_simple_paths(staged_tree, source='s0', target=staged_tree.leaves)
    pairs=list()
    for path in paths:
        path = [int(elem[1:]) for elem in path]
        temp_pairs=(zip(path[:-1],path[1:]))
        for temp_pair in temp_pairs:
            pairs.append(list(temp_pair))
    pairs.sort()
    pairs_S=[str(pair) for pair in pairs]
    from collections import Counter
    prior_vals=list(Counter(pairs_S).values())
    i=0
    SE_priors=staged_tree._create_default_prior(1)
    for edges in range(0,len(SE_priors)):
        for j in range(0,len(SE_priors[edges])):
            SE_priors[edges][j]=prior_vals[i]*alpha
            i=i+1
    return SE_priors
# %%
def get_hyperstage(st,prior):
    staged_tree = copy.deepcopy(st)
    staged_tree._store_params(prior=prior, alpha=None,hyperstage=None)
    hyperstage = staged_tree._create_default_hyperstage()
    new_hyperstage=[]
    if max([len(x) for x in staged_tree.posterior_list]) > 2:
        raise error
    else:
        for hyperset in hyperstage:
            if len(hyperset) == 1:
                new_hyperset=[hyperset]
            else:
                sit_in_hyperset = [sit in hyperset for sit in staged_tree.situations]
                hyperset_posts = [staged_tree.posterior_list[x] for x in range(0,len(staged_tree.posterior_list)) if sit_in_hyperset[x]]
                hyperset_posts=np.array(hyperset_posts)

                if hyperset_posts.ndim == 1:
                    new_hyperset=[hyperset]
                else:
                    independent_posterior_means = hyperset_posts/np.sum(hyperset_posts,axis=1)[:,None]
                    independent_posterior_means = independent_posterior_means[:,0]# only first value
                    sorted_vals = independent_posterior_means.argsort()
                    hyperset_sorted = [hyperset[i] for i in sorted_vals]
                    first = hyperset_sorted[0:-1]
                    second = hyperset_sorted[1:]
                    new_hyperset=list(zip(first,second))
                    new_hyperset = [list(x) for x in new_hyperset]
            new_hyperstage = new_hyperstage + new_hyperset
    return(new_hyperstage)

# %%

def get_hyperstage_sample_means(st):
    staged_tree = copy.deepcopy(st)
    staged_tree._store_params(prior=None, alpha=None,hyperstage=None)
    hyperstage = staged_tree._create_default_hyperstage()
    new_hyperstage=[]
    if max([len(x) for x in staged_tree.posterior_list]) > 2:
        raise error
    else:
        for hyperset in hyperstage:
            if len(hyperset) == 1:
                new_hyperset=[hyperset]
            else:
                sit_in_hyperset = [sit in hyperset for sit in staged_tree.situations]
                hyperset_posts = [staged_tree.posterior_list[x] for x in range(0,len(staged_tree.posterior_list)) if sit_in_hyperset[x]]
                hyperset_posts = np.array(hyperset_posts)
                hyperset_prior = [staged_tree.prior_list[x] for x in range(0,len(staged_tree.posterior_list)) if sit_in_hyperset[x]]
                hyperset_prior = np.array(hyperset_prior)
                #when both elements of hyperset_prior are equal to hyperset_posts add one to hyperset_posts

                # hyperset_posts[hyperset_prior==hyperset_posts] = hyperset_posts[hyperset_prior==hyperset_posts] + 1
                hyperset_sample = hyperset_posts - hyperset_prior
                #replace rows in a hyperset_sample that are all zeros with all ones
                hyperset_sample[hyperset_sample.sum(axis=1)==0] = 1

                if hyperset_sample.ndim == 1:
                    new_hyperset=[hyperset]
                else:
                    independent_posterior_means = hyperset_sample/np.sum(hyperset_sample,axis=1)[:,None]
                    independent_posterior_means = independent_posterior_means[:,0]# only first value
                    sorted_vals = independent_posterior_means.argsort()
                    hyperset_sorted = [hyperset[i] for i in sorted_vals]
                    first = hyperset_sorted[0:-1]
                    second = hyperset_sorted[1:]
                    new_hyperset=list(zip(first,second))
                    new_hyperset = [list(x) for x in new_hyperset]
            new_hyperstage = new_hyperstage + new_hyperset
    return(new_hyperstage)

# %%
