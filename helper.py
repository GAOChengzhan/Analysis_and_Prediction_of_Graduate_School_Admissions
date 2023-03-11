# ## Naive Bayes Analysis
# Import Libraries
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import numpy as np
from pandas.api.types import CategoricalDtype
import re
# Data is stored in external directory
path = "../grad_cafe_admissions.csv"
fields = []
rows = []
 
# Import data
# reading csv file
with open(path, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
# Convert into dataframe
df = pd.DataFrame (rows, columns = fields)

nan_count=list()
for field in fields:
    nan_count.append(df[df[field]=='NA'][field].count())

# Cleaning dataset
def remove_non_latin(string):
    string = re.sub(r'[^\x41-\x5a|\x61-\x7a]',r'', string)
    string = re.findall('[A-Z][^A-Z]*', string)
    
    return ' '.join(string)
for field in ["institution", "major", "degree", "sem"]:
    df[field] = df[field].apply(remove_non_latin)
df.head()

# ### 2.C Training Using Gaussian Naive Bayes
# One-hot vector
def get_ohv(Z_original, Z_, f):
    Z_temp = np.array(Z_original)
    Z_ohv = np.zeros((Z_temp[:,f].size, Z_[:,f].max()+1))
    
    Z_ohv[np.arange(Z_temp[:,f].size), Z_temp[:,f]]=1
    return Z_ohv

# #### A. Training for Data with no-missing labels
# Remove rows with NAN object in any rows
df_new = df
df_imputed = df_new[(df_new["gpa"].str.isnumeric()==True) & (df_new["gre_v"].str.isnumeric()==True) &(df_new["gre_q"].str.isnumeric()==True) & (df_new["gre_w"].str.isnumeric()==True) & (df_new["gre_subject"].str.isnumeric()==True) ]
# Remove Field values beyond permissble limits
# Mean is used to fill empty cells
for i, field in enumerate(fields[-5:]):
    df_imputed.loc[(df_imputed[field].str.isnumeric()==False) & (df_imputed[field].to_numpy().astype(float)<=field_max[i]) &(df_imputed[field].to_numpy().astype(float)>=field_min[i]), field]=field_mean[field]
df_imputed.head()

# Verify dataset. For this dataset only two output labels are considered "Accepted" and "Rejceted"
cats = ["Accepted", "Rejected"]
cat_type = CategoricalDtype(categories=cats, ordered=False)
X = df_imputed[fields[-5:]].to_numpy()
Y_cat = df_imputed[fields[-9]].astype(cat_type)
Y = Y_cat.cat.codes.to_numpy() + 1
Y_labels = Y_cat.cat.categories
Z = list()
Z_label = list()
for field in ["institution", "major", "degree", "sem"]:
    Z_cat = df_imputed[field].astype(CategoricalDtype())
    Z.append(Z_cat.cat.codes.to_numpy())
    Z_label.append(Z_cat.cat.categories)
Z = np.array(Z).T
X.shape, Y.shape, Y_labels, Z.shape
    
# K-fold cross Validation
k = 10
# prepare cross validation
kf = KFold(n_splits=k)
# enumerate splits
Score = list()
for i, (train_index, test_index) in enumerate(kf.split(X)):
    clf = GaussianNB()
    X_train = X[train_index][:,:-2]
    for k in range(len(Z_label)):
        X_train = np.hstack((X_train, get_ohv(Z[train_index],Z, k)))
    clf.fit(X_train, Y[train_index])
    X_test = X[test_index][:,:-2]
    for k in range(len(Z_label)):
        X_test = np.hstack((X_test, get_ohv(Z[test_index],Z, k)))
    Score.append(clf.score(X_test, Y[test_index]))
print("Training using Gaussian Naive Bayes(full input vector)")
print(u"Accuracy after 10-fold cross-validation {0:.3f} \u00B1 {1:.3f}%".format(100*np.mean(Score), 100*np.std(Score)))
print("Max accuracy {0:.3f}%".format(np.max(Score)*100))

