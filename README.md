## Algorithm for co-elution data profiling using machine learning.
#### Based on elements of the MATLAB algorithm from Stacey et al. 2017, BMC Bioinformatics, rewritten in Python with changes
#### Author: Piotr Grabowski, 2018, Rappsilber group

#### Overall structure:
1. Read data
2. Filter profiles (with minimum number of consecutive data points and intensity)
3. Impute single missing values with 2 neighboring datapoints (using their mean), remaining data set to 0
4. Smooth profiles
5. Fit Gaussian mixture models to each profile (consider using L1 norm robust fitting to decrease outliers' effects). Perform automatic model selection using AIC/BIC metrics.
6. Feature engineering: Pearson correlation coefficient (**test others, MAD-based**), PCC pval, Euclidean distance between profiles, peak location (check paper) and Co-apex score (check paper).
7. Unsupervised and supervised machine learning on the features using scikit-learn

#### Data pre-requisites:
1. iBAQ values
2. Replicates separate

#### To do:
1. Dump requirements.txt
2. Add tests
