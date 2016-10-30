import lccm
import numpy as np
import pandas as pd
import pylogit
import warnings
from collections import OrderedDict

# Load the data file
inputFilePath = 'data/'
inputFileName = 'TrainingData.txt'

print '\nReading %s' %inputFileName 
data = np.loadtxt(open(inputFilePath + inputFileName, 'rb'), delimiter='\t')

# Convert to a pandas dataframe
df = pd.DataFrame(data, columns=['indID', 'altID', 'obsID', 'choice', 'zipInd',
        'hhIncome', 'male', 'adopters', 'stationDummy', 'googleDummy', 'accessibility'])

# Clean up and scale variables as needed
df['hhIncome'] = df.hhIncome / 1000


# Class-membership model: 
# The first step is to specify the number of latent classes

nClasses = 3

# accounting for weights - choice-based sampling (Moshe & Lerman)
# Weighted Exogenous Sample Maximum Likelihood (WESML)
weightAdopters = 0.0003853/0.404
weightNonAdopters = 0.9996147/0.596    
indWeightsA = np.repeat(weightAdopters, 300)
indWeightsNA = np.repeat(weightNonAdopters,201 )        
indWeights = np.hstack((indWeightsA,indWeightsNA))  

# Utility for each of the latent classes is specified by creating matrices 
# expVarsClassMem, of size (nExpVars x nRows). The (i, j)th element of the matrix denotes 
# the ith explanatory variable entering the utility for the decision-maker corresponding 
# to the jth row in the data file.

class_membership_spec = ['hhIncome', 'male']
class_membership_labels = ['Class-specific constant','Monthly Income (1000s $)', 'male' ]


# UTILITY SPECIFICATIONS FOR EACH LATENT CLASS

# New vars: carshare choice indicator, and interaction of utility components with choice

df['altcarshare'] = (df.altID == 1).astype(int)
df['v_accessibility'] = df.altID * df.accessibility
df['v_adopters'] = df.altID * df.adopters
df['v_google_dummy'] = df.altID * df.googleDummy


# Set up class-specific specifications in a pylogit-style format, and corresponding labels

class_specific_specs = [
	OrderedDict([
		('altcarshare', [1]), 
		('v_accessibility', [1]), 
		('v_google_dummy', [1]) ]),
	OrderedDict([
		('altcarshare', [1]),
		('v_accessibility', [1]),
		('v_adopters', [1]),
		('v_google_dummy', [1]) ]),
	OrderedDict([
		('altcarshare', [1])])
]

class_specific_labels = [
	['ASC (CarShare)','Accessibility', 'Google Employee'],
	['ASC (CarShare)', 'Accessibility', 'Cumulative Adopters (t-1)', 'Google Employee'],
	['ASC (CarShare)']
]


# Invoke the parameter estimation

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    lccm.lccm_fit(data = df,
                  ind_id_col = 'indID', 
                  obs_id_col = 'obsID',
                  alt_id_col = 'altID',
                  choice_col = 'choice', 
                  nClasses = nClasses, 
                  class_membership_spec = class_membership_spec,
                  class_membership_labels = class_membership_labels,
                  class_specific_specs = class_specific_specs,
                  class_specific_labels = class_specific_labels, 
                  indWeights = indWeights)


