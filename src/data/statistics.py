import pandas as pd

import src.data.utils.helper_functions as helper_functions
import logging
import os



#%% get logger to work
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
filename = "all_methods_train"

window_size = 8
# get logger
logger = logging.getLogger(__name__)
logger.info(filename)

#%% load data

data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                           filename + '.json')  # get decoded path

#%%

df = pd.read_json("all_methods_train.json", orient='records')


#%% preproc
# some basic operations: preprocessing parameters
df['parameters'] = df['parameters'].apply(helper_functions.turn_all_to_lower)
df['parameters'] = df['parameters'].apply(helper_functions.split_params)


#%% preproc methodbody
# some basic operations: preprocessing method body
df['methodBody'] = df['methodBody'].apply(helper_functions.removeOptional)


#%% preproc

df['methodBody'] = df['methodBody'].apply(helper_functions.replace_string_values)
#%% preproc

df["methodBody"] = df['methodBody'].apply(helper_functions.turn_strings_to_list)
#%% preproc

df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)

#%% preproc

df["methodBody"] = df['methodBody'].apply(helper_functions.turn_all_to_lower)

#%%

#clean from function structure
df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)

#concat type, params, method body
df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]

# df['methodName']= df['methodName'].str.lower() should a function be all lower?

#%%
# compute some statistics
df['methodBodyLength'] = df['methodBody'].apply(helper_functions.compute_col_length)

#%%

#create copy of df
df_mod = df.copy()

#%%

#turn list in rows to strings
df_mod['methodBody'] = df_mod['methodBody'].apply(lambda x: " ".join(x))

#remove some rows
df_mod = df_mod[df_mod['methodBody'] != 'empt']
df_mod = df_mod[df_mod['methodBody'] != '{ }']

#turn back to list
df_mod['methodBody'] = df_mod['methodBody'].apply(lambda x: x.split())

#%%

#split list in rows to strings
df_mod['methodBodyCleaned'] = df_mod['methodBodyCleaned'].apply(lambda x: " ".join(x))

df_mod = df_mod[df_mod['methodBodyCleaned'] != ' ']
df_mod = df_mod[df_mod['methodBodyCleaned'] != '']
#back to list
df_mod['methodBodyCleaned'] = df_mod['methodBodyCleaned'].apply(lambda x: x.split())


#%%
#compute data distribution of empty functions excluded 1) without spec chars, 2) with special chars

len_method_body_excl_empt_excl_spec = df_mod['methodBodyCleaned'].apply(helper_functions.compute_col_length)
len_method_body_excl_empt_excl_spec = len_method_body_excl_empt_excl_spec.describe()

len_method_body_excl_empt_incl_spec = df_mod['methodBodyLength'].describe()

print("len_method_body_excl_empt_excl_spec \n{}".format(len_method_body_excl_empt_excl_spec))
print("len_method_body_excl_empt_incl_spec \n{}".format(len_method_body_excl_empt_incl_spec))

#%%

len_method_body_incl_empt_excl_spec = df['methodBodyCleaned'].apply(helper_functions.compute_col_length)
len_method_body_incl_empt_excl_spec = len_method_body_incl_empt_excl_spec.describe()

len_method_body_incl_empt_incl_spec = df['methodBodyLength'].describe()
print("len_method_body_incl_empt_excl_spec \n{}".format(len_method_body_incl_empt_excl_spec))
print("len_method_body_incl_empt_incl_spec \n{}".format(len_method_body_incl_empt_incl_spec))


'''

#%%

df_where_is_empty = df_mod.copy()
df_where_is_empty = df_where_is_empty[df_where_is_empty['methodBodyLength'] == 1]

#evtl TODO df.dropna

#%%
import matplotlib.pyplot as plt

plt.boxplot([df_mod['methodBodyLength'],
             df['methodBodyLength'],
             len_method_body_incl_empt_excl_spec,
             len_method_body_excl_empt_excl_spec])
plt.xticks([1, 2, 3, 4], ['Incl. Empty \n Functions & \n incl. Special Chars',
                       'Excl. Empty \n Functions & \n incl. Special Chars',
                       'Incl. Empty \n Functions & \n excl. Special Chars',
                       'Excl. Empty \n Functions & \n excl. Special Chars'])
plt.show()



#%%

#compute avg length of params
avg_mean_params = df['parameters'].apply(helper_functions.compute_col_length)

#compute avg length of body
avg_mean_body = df['methodBodyCleaned'].apply(helper_functions.compute_col_length)

# compute avg length of body + params + type
avg_mean_body_params_type = df['concatMethodBodyCleaned'].apply(helper_functions.compute_col_length)

'''