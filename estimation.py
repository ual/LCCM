import lccm
import numpy as np
import warnings

# Load the data file

inputFilePath = 'data/'
inputFileName = 'TrainingData.txt'

print '\nReading %s' %inputFileName 
data = np.loadtxt(open(inputFilePath + inputFileName, 'rb'), delimiter='\t')


# Class-membership model: 
# The first step is to specify the number of latent classes and to identify the column 
# in the data file denoting the individual IDs. 

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


hhIncome = data[:, 5]
hhIncome = hhIncome / 1000

male = (data[:, 6] == 1).astype(int)

expVarsClassMem = np.vstack((np.ones(data.shape[0]),hhIncome, male))

namesExpVarsClassMem = ['Class-specific constant','Monthly Income (1000s $)', 'male' ]
        

# Constraints on available latent classes are imposed through the variable availClasses, 
# a 2D array of size (nClasses x nRows), where the (i,j)th element equals 1 if the ith 
# latent class is available to the decision-maker corresponding to the jth row in the 
# dataset, and 0 otherwise.

availIndClasses = np.vstack((np.logical_not(hhIncome < 0).astype(int), 
                          np.logical_not(hhIncome < 0).astype(int),np.logical_not(hhIncome < 0).astype(int)))


# Class-specific choice model
# The first step is to identify key columns in the data file denoting the individual,
# observation, alternatives and choices.

indID = data[:, 0]    
obsID = data[:, 2]
altID = data[:, 1]
choice = np.reshape(data[:, 3], (data.shape[0], 1))

# Choice set constraints are imposed through the variable availAlts, a list of 
# size nClasses, where the sth element is an array containing identifiers for the 
# alternatives that are available to decision-makers belonging to the sth class.
    
availAlts = [np.array([0, 1]), 
                np.array([0, 1]), np.array([0, 1])]    

# Utility for each of the classes is specified by creating a list expVarsClassSpec, of size 
# nClasses, where the sth element is a matrix of size (nExpVars x nRows) containing
# the explanatory variables entering the class-specific utilities for the sth 
# latent class. The (i, j)th element of the matrix denotes the ith explanatory 
# variable entering the utility for the alternative corresponding to the jth row 
# in the data file.

expVarsClassSpec, namesExpVarsClassSpec = [], []

altcarshare = (altID == 1).astype(int)
altnocarshare = (altID == 0).astype(int)  

zipInd = data[:,4]
googleDummy = data[:,9]


stationDummy = data[:,8]
adopters = data[:,7]

accessibility = data[:,10]    

stationDummyNeeded = altID * stationDummy
adoptersNeeded = altID * adopters
googleDummyNeeded = altID * googleDummy 
accessibilityNeeded = altID * accessibility

expVarsClassSpec.append(np.vstack(( altcarshare, accessibilityNeeded,  googleDummyNeeded)))    
namesExpVarsClassSpec.append(['ASC (CarShare)','Accessibility', 'Google Employee'])

expVarsClassSpec.append(np.vstack((altcarshare,  accessibilityNeeded, adoptersNeeded, googleDummyNeeded)))  
namesExpVarsClassSpec.append(['ASC (CarShare)', 'Accessibility', 'Cumulative Adopters (t-1)', 'Google Employee']) 

        
expVarsClassSpec.append(np.vstack((altcarshare)).transpose())   
namesExpVarsClassSpec.append(['ASC (CarShare)'])
        

# Parameter Estimation

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lccm.lccm_fit(nClasses, 
            indID, expVarsClassMem, namesExpVarsClassMem, availIndClasses,
            obsID, altID, choice, availAlts, expVarsClassSpec, namesExpVarsClassSpec, indWeights)


