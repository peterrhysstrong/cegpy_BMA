# %%
from random import seed
from Bayesian_model_averaging_CEG import *
import pandas as pd
from src.cegpy.trees import event, staged
from src.cegpy.graphs import ceg
import copy
import matplotlib.pyplot as plt

data = pd.read_csv("datasets/Falls_Data_10000.csv")
# create event tree
show_event_tree(data)

# %%

# create stage tree
st = staged.StagedTree(data)
hyperstage = st._create_default_hyperstage()
HAC_st = copy.deepcopy(st)
# run HAC and show CEG of output 
HAC_st.calculate_AHC_transitions( alpha=4, hyperstage = hyperstage)
HAC_CEG = ceg.ChainEventGraph(HAC_st)
HAC_CEG.create_figure()

#%%

# run BMA with w-hac for each hyperset
staging_output, model_weights,loglikelihoods = Bayesian_model_averaging_CEGs(st, K_max=100, prior_weight=4, hyperstage = hyperstage)

#%%

# get the staging for the whole hyperstage
full_staging_output, full_model_weights = get_full_model_weights(st, model_weights, staging_output)
# plot model weights for the well performing models
y_pos = np.arange(len(full_staging_output))
if len(full_staging_output)>1:
    # full_model_weights = full_model_weights.flatten()
    sorted_model_weights = sorted(full_model_weights,reverse=True)
labels=["M"+str(i+1) for i in range(0,len(full_staging_output))]
plt.bar(labels,sorted_model_weights , align='center', alpha=0.5)
plt.ylabel('Model weights')
plt.xticks(rotation=90)
plt.show()

#%%
# calculate well performing unions and intersections
intersection_staging = get_staging_intersection(staging_output)
union_staging = get_staging_union(staging_output)
intersection_set = []
union_set = []
for sets in list(intersection_staging.values()):
    intersection_set += sets
for sets in list(union_staging.values()):
    union_set += sets
#%%
# save figures in high dpi
# %%
# show ceg of intersection set
coarsest=copy.deepcopy(st)
coarsest.calculate_full_transitions(hyperstage=intersection_set)
coarsest.create_figure()

coarsest_ceg=ceg.ChainEventGraph(coarsest)
coarsest_ceg.create_figure()

#%%
# show ceg of union set
coarsest=copy.deepcopy(st)
coarsest.calculate_full_transitions(hyperstage=union_set)
coarsest.create_figure()

coarsest_ceg=ceg.ChainEventGraph(coarsest)
coarsest_ceg.create_figure()

