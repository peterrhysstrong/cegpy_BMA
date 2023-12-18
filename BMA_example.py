# %%
from random import seed
from Bayesian_model_averaging_CEG import *
import pandas as pd
from src.cegpy.trees import event, staged
from src.cegpy.graphs import ceg
import copy
import matplotlib.pyplot as plt


# create event tree
data = pd.read_csv("datasets/Falls_Data.csv")
#to test on smaller dataset try
# data = pd.read_csv("datasets/Falls_Data_10000.csv")
show_event_tree(data)


# %%

# create stage tree
st = staged.StagedTree(data)
#for any stuctured missing data use the optional argument: struct_missing_label= ""
hyperstage = st._create_default_hyperstage()
HAC_st = copy.deepcopy(st)
#%%
# run HAC and show CEG of output 
HAC_st.calculate_AHC_transitions(alpha=4, hyperstage = hyperstage)
HAC_CEG = ceg.ChainEventGraph(HAC_st)
HAC_CEG.create_figure()

#%%
# run BMA with w-hac for each hyperset
staging_output, model_weights,loglikelihoods = Bayesian_model_averaging_CEGs(st, K_max=100, prior_weight=4, hyperstage = hyperstage)

#%%

#plot each hyperset
for h in range(len(staging_output)):
    if len(model_weights[h])>1:
        labels=[str(x) for x in staging_output[h]]
        data = model_weights[h].flatten() 

        plt.pie(data, labels=labels)
        plt.title( "Normalised Bayes factor ratios for hyperset\n"+ str(hyperstage[h]))        
        plt.show()
#%%
for h in range(len(staging_output)):
    if len(model_weights[h])>1:
        labels=[str(x) for x in staging_output[h]]
        data = model_weights[h].flatten()  

        order=sorted(range(len(data)), key=data.__getitem__,reverse=True)
        ordered_labs=[labels[i] for i in order] 
        ordered_data=[data[i] for i in order]

        plt.bar(ordered_labs,ordered_data , align='center', alpha=0.5)
        plt.title( "Normalised Bayes factor ratios for hyperset\n"+ str(hyperstage[h]))        
        plt.ylabel('Model weights')
        plt.xticks(rotation=90)
        plt.show()
#%%

# get the staging for the whole hyperstage
full_staging_output, full_model_weights = get_full_model_weights(model_weights, staging_output)
# plot model weights for the well performing models
y_pos = np.arange(len(full_staging_output))
if len(full_staging_output)>1:
    sorted_model_weights = sorted(full_model_weights,reverse=True)
else:
    sorted_model_weights = full_model_weights
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

# %%
