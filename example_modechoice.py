import lccm
import numpy as np
import pandas as pd
import pylogit
from collections import OrderedDict

df = pd.DataFrame.from_csv('data/valueOfGreen.dat', sep='\t', index_col=None)

# Reshape

ind_vars = ['ID', 'GENDER', 'VEG', 'CAR_OWNER', 'INCOME']

alt_varying_vars = {'travel_time': dict([(1, 'TT_A1'), (2, 'TT_A2'), (3, 'TT_B1'), 
                                (4, 'TT_B2'), (5, 'TT_T'), (6, 'TT_K'), (7, 'TT_W')]),
                    'travel_cost': dict([(1, 'C_A1'), (2, 'C_A2'), (3, 'C_B1'),
                                (4, 'C_B2'), (5, 'C_T')]),
                    'emissions': dict([(1, 'GG_A1'), (2, 'GG_A2'), (3, 'GG_B1'), 
                                (4, 'GG_B2'), (5, 'GG_T')])}

availability_vars = {1: 'A1_AV', 2: 'A2_AV', 3: 'B1_AV', 4: 'B2_AV', 5: 'T_AV', 6: 'K_AV', 7: 'W_AV'}

alt_id_col = 'ALT_ID'

# 'ID' is the decision-maker, but there are multiple observations (choice scenarios) for each
df['OBS_ID'] = np.arange(df.shape[0], dtype=int) + 1
obs_id_col = 'OBS_ID'

choice_col = 'CHOICE'

data = pylogit.convert_wide_to_long(df, ind_vars, alt_varying_vars, 
                availability_vars, obs_id_col, choice_col, new_alt_id_name=alt_id_col)


# Specify important columns

ind_id_col = 'ID'
obs_id_col = 'OBS_ID'
alt_id_col = 'ALT_ID'
choice_col = 'CHOICE'


# base case is alt_id = 7 (walking)

spec =  OrderedDict([
            ('intercept', [1, 2, 3, 4, 5, 6]),
            ('travel_time', [[1, 2, 3, 4, 5, 6, 7]]),
            ('travel_cost', [[1, 2, 3, 4, 5, 6, 7]]),
            ('emissions', [[1, 2, 3, 4, 5, 6, 7]])
        ])

labels = OrderedDict([
            ('intercept', ['asc_drive1', 'asc_drive1', 'asc_bus1', 'asc_bus2', 'asc_train', 'asc_bike']),
            ('travel_time', ['travel time']), 
            ('travel_cost', ['travel cost']),
            ('emissions', ['emissions'])
        ])


# Class membership model

n_classes = 2

class_membership_spec = ['intercept', 'GENDER', 'VEG', 'INCOME', 'CAR_OWNER']
class_membership_labels = ['ASC', 'Gender', 'Vegetarian', 'Income', 'Car owner']

# Starting values for class membership

value = 0
length = len(class_membership_spec) * (n_classes -1)
paramClassMem = np.array([value] * length)


# Class-specific choice model

class_specific_specs = [spec, spec]
class_specific_labels = [labels, labels]

lccm.lccm_fit(data = data,
              ind_id_col = ind_id_col,
              obs_id_col = obs_id_col,
              alt_id_col = alt_id_col,
              choice_col = choice_col,
              n_classes = n_classes,
              class_membership_spec = class_membership_spec,
              class_membership_labels = class_membership_labels,
              class_specific_specs = class_specific_specs,
              class_specific_labels = class_specific_labels,
              paramClassMem = paramClassMem)


class_specific_labels = [labels, labels]