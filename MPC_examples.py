# %%
from distutils.log import error
from random import seed
from Bayesian_model_averaging_CEG import *
from Mean_posterior_search import *
import pandas as pd
from src.cegpy.trees import event, staged
from src.cegpy.graphs import ceg
import copy
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial import distance

#%%
data = pd.read_csv("datasets/BN_df2.csv")


for i in range(0,len(data.columns)):
    	data[data.columns[i]]=data[data.columns[i]].astype(str)+"_"+str(i+1)

# show_event_tree(data)

#%%
# full_st = staged.StagedTree(data)
# # run HAC and show CEG of output 
# non_bin_prior = get_priors(full_st)
# st = time.time()
# full_st.calculate_AHC_transitions(prior = non_bin_prior)
# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')
# print(full_st.ahc_output)
# full_st.create_figure()

# full_CEG = ceg.ChainEventGraph(full_st)
# full_CEG.create_figure()

#%%

data = make_binary(data)
# show_event_tree(data)

#%%
# data = pd.read_csv("datasets/df_ceg.csv")
# data = data.iloc[:,4:]
# create event tree
# data = data.iloc[:,:3]
# data = pd.read_csv("datasets/Moves.csv")
# data = pd.read_csv("datasets/missing_moves.csv")
# data = pd.read_csv("datasets/missing+moves.csv")
# data = pd.read_csv("datasets/no_missing_moves.csv")
# data = pd.read_csv("datasets/Missing_as_no_moves.csv")


# data = data.iloc[:,1:]


# %%

MPC_st = staged.StagedTree(data)
# run HAC and show CEG of output 
bin_prior = get_priors(MPC_st)
# bin_prior=[[12,12],[6,6],[6,6],[3,3],[3,3],[3,3],[3,3],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
new_hyperstage = get_hyperstage(MPC_st)

st = time.time()
MPC_st.calculate_MPC_transitions(hyperstage = new_hyperstage,prior = bin_prior)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print(MPC_st.ahc_output)
MPC_st.create_figure()

HAC_CEG = ceg.ChainEventGraph(MPC_st)
HAC_CEG.create_figure()

# %%

AHC_st =  staged.StagedTree(data)
st = time.time()
AHC_st.calculate_AHC_transitions(prior = bin_prior)
et = time.time()
elapsed_time = et - st

print('Execution time:', elapsed_time, 'seconds')
print(AHC_st.ahc_output)

AHC_st.create_figure()
# CEG = ceg.ChainEventGraph(AHC_st)
# CEG.create_figure()

# %%
diff=MPC_st.ahc_output['Loglikelihood']-AHC_st.ahc_output['Loglikelihood']
print('logBF posterior compared to AHC',diff)
# %%


# def paired_hyperstage_setting(hyperstage,sort=None):
#     hyperstage = [list(x) for x in hyperstage]
#     new_hyperstage = []
#     for hyperset in hyperstage:
#         if len(hyperset)==1:
#             new_hyperset=[hyperset]
#         else:
#             first=[hyperset[0]]*(len(hyperset)-1)
#             second=hyperset[1:]
#             new_hyperset=list(zip(first,second))
#             new_hyperset = [list(x) for x in new_hyperset]
#         new_hyperstage = new_hyperstage + new_hyperset
#     return new_hyperstage

# %%
# first=copy.deepcopy(staged_tree)
# fir_hyper = paired_hyperstage_setting(AHC_st.ahc_output['Merged Situations'])

# first.calculate_full_transitions(hyperstage=fir_hyper)
# print(first.ahc_output)
# CEG = ceg.ChainEventGraph(first)
# CEG.create_figure()

# # %%
# second=copy.deepcopy(staged_tree)
# sec_hyper = paired_hyperstage_setting(MPC_st.ahc_output['Merged Situations'])
# second.calculate_full_transitions(hyperstage=sec_hyper)
# print(second.ahc_output)
# second.create_figure()
# CEG = ceg.ChainEventGraph(second)
# CEG.create_figure()
# #%%
# first.ahc_output['Loglikelihood']-second.ahc_output['Loglikelihood']

# %%
# posterior_array=np.array(staged_tree.posterior_list)
# independent_posterior_means=posterior_array/np.sum(posterior_array,axis=1)[:,None]
# plt.scatter(independent_posterior_means[:,0],independent_posterior_means[:,1])
# plt.show()

# # %%
# independent_posterior_means_float=independent_posterior_means.astype(float)
# distances = distance.cdist(independent_posterior_means_float, independent_posterior_means_float, 'euclidean')
# plt.imshow(distances)
# plt.show()
# %%


