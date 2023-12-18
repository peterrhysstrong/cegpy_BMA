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

alpha=1
data = pd.read_csv("datasets/CHDS.latentexample1.csv")
data = data[['Social','Economic','Admission','Events']]
# data = data[['Social','Economic','Events','Admission']]

# data= pd.read_csv("datasets_test/selfy.csv")
for i in range(0,len(data.columns)):
    	data[data.columns[i]]=data[data.columns[i]].astype(str)+"_"+str(i+1)

#%%
full_st = staged.StagedTree(data)
# normal AHC search on tree

non_bin_prior=get_priors(full_st, alpha)

st = time.time()
full_st.calculate_AHC_transitions(prior = non_bin_prior)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print(full_st.ahc_output)
full_st.create_figure()

full_CEG = ceg.ChainEventGraph(full_st)
full_CEG.create_figure()

#%%
def make_binary_CHDS(df,order):
    df_bin = pd.DataFrame()
    for col in df.columns:
        # df_bin_temp = pd.DataFrame()
        #uni = df[col].unique()
        uni = df[col].unique()[::-1]
        if len(uni) == 2:
            df_bin[str(col) + "_" + str(uni[0])] = df[col].apply(lambda x: uni[0] if x == uni[0] else uni[1])
        else:
            uni = order
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

# data_other = make_binary_CHDS(data,['Average_4',"High_4",'Low_4'])

# MPC_st = staged.StagedTree(data_other)
# bin_prior=get_priors(MPC_st)
# new_hyperstage= get_hyperstage(MPC_st)


# st = time.time()
# MPC_st.calculate_MPC_transitions(hyperstage = new_hyperstage,prior = bin_prior)
# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')
# print(MPC_st.ahc_output)
# MPC_st.create_figure()

# HAC_CEG = ceg.ChainEventGraph(MPC_st)
# HAC_CEG.create_figure()

# # %%

# AHC_st = staged.StagedTree(data_other)
# st = time.time()
# AHC_st.calculate_AHC_transitions(prior = bin_prior)
# et = time.time()
# elapsed_time = et - st

# print('Execution time:', elapsed_time, 'seconds')
# print(AHC_st.ahc_output)

# AHC_st.create_figure()
# CEG = ceg.ChainEventGraph(AHC_st)
# CEG.create_figure()

# # %%
# diff=MPC_st.ahc_output['Loglikelihood']-AHC_st.ahc_output['Loglikelihood']
# print('logBF posterior compared to AHC',diff)
# %%


# data_low = make_binary_CHDS(data,['Low_4','Average_4',"High_4"])

# # %%

# MPC_st = staged.StagedTree(data_low)
# new_hyperstage = get_hyperstage(MPC_st)
# st = time.time()
# MPC_st.calculate_MPC_transitions(hyperstage = new_hyperstage,prior = bin_prior)
# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')
# print(MPC_st.ahc_output)
# MPC_st.create_figure()

# HAC_CEG = ceg.ChainEventGraph(MPC_st)
# HAC_CEG.create_figure()

# # %%

# AHC_st = staged.StagedTree(data_low)
# hyperstage = AHC_st._create_default_hyperstage()

# st = time.time()
# AHC_st.calculate_AHC_transitions(hyperstage=hyperstage,prior = bin_prior)
# et = time.time()
# elapsed_time = et - st

# print('Execution time:', elapsed_time, 'seconds')
# print(AHC_st.ahc_output)

# AHC_st.create_figure()
# CEG = ceg.ChainEventGraph(AHC_st)
# CEG.create_figure()

# # %%
# diff=MPC_st.ahc_output['Loglikelihood']-AHC_st.ahc_output['Loglikelihood']
# print('logBF posterior compared to AHC',diff)

#%%
 

data_high = make_binary_CHDS(data,["High_4",'Average_4','Low_4'])

# %%

MPC_st = staged.StagedTree(data_high)
new_hyperstage = get_hyperstage(MPC_st)
bin_prior=get_priors(MPC_st,alpha)
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

AHC_st = staged.StagedTree(data_high)
hyperstage = AHC_st._create_default_hyperstage()

st = time.time()
AHC_st.calculate_AHC_transitions(hyperstage=hyperstage,prior = bin_prior)
et = time.time()
elapsed_time = et - st

print('Execution time:', elapsed_time, 'seconds')
print(AHC_st.# run HAC and show CEG of output 
ahc_output)

AHC_st.create_figure()
CEG = ceg.ChainEventGraph(AHC_st)
CEG.create_figure()

# %%
diff=MPC_st.ahc_output['Loglikelihood']-AHC_st.ahc_output['Loglikelihood']
print('logBF posterior compared to AHC',diff)

#%%
# non binary CEG fit based on bianry learning
aprox_full_st = staged.StagedTree(data)
non_bin_prior=[[12,12],[6,6],[6,6],[3,3],[3,3],[3,3],[3,3],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]

aprox_hyperstage =[['s0'],
 ['s1', 's2'],
 ['s3', 's4'],['s5','s6'],
 ['s7'], ['s8'],['s9', 's11'],['s12'], ['s10', 's13', 's14']]
st = time.time()
aprox_full_st.calculate_AHC_transitions(prior = non_bin_prior,hyperstage=aprox_hyperstage)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print(aprox_full_st.ahc_output)
aprox_full_st.create_figure()

aprox_full_CEG = ceg.ChainEventGraph(aprox_full_st)
aprox_full_CEG.create_figure()
# %%

# best_st = staged.StagedTree(data_high)

# bin_prior=[[12,12],[6,6],[6,6],[3,3],[3,3],[3,3],[3,3],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]

# good_hyperstage=[
#     ['s8', 's9', 's11'],
#   ['s16', 's24', 's26', 's20'],
#   ['s28', 's22', 's18', 's30'],
#   ['s3', 's4'], ['s5','s6'],
#   ['s12', 's13', 's10', 's14'],
#   ['s0',],
#   ['s1'],
#   ['s2'],
#   ['s7']
#   ]# data_high = make_binary_CHDS(data,["High_3",'Average_3','Low_3'])

# st = time.time()
# best_st.calculate_full_transitions(hyperstage=good_hyperstage,prior = bin_prior)
# et = time.time()
# elapsed_time = et - st

# print('Execution time:', elapsed_time, 'seconds')
# print(best_st.ahc_output)

# best_st.create_figure()

# %%
