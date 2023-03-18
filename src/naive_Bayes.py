import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import numpy as np
from pandas.api.types import CategoricalDtype
import re
import os

def import_data(path, isGradCafe=True):
    """
        Import data from csv into dataframe
        Returns
            pandas.Dataframe
    """
    assert isinstance(path, str), "Not a string!"
    if isGradCafe:
        fields = []
        rows = []
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            fields = next(csvreader)

            for row in csvreader:
                rows.append(row)
        df = pd.DataFrame (rows, columns = fields)
    else:
        df = pd.read_csv(path)
    
    return df

def clean_df(df):
    """
    Clean the data
    Returns
        pandas.DataFrame
    """
    assert isinstance(df, pd.DataFrame), "Not a dataframe!"
    nan_count=list()
    fields = df.columns.values.tolist()
    for field in fields:
        nan_count.append(df[df[field]=='NA'][field].count())

    def remove_non_latin(string):
        """
        Remove non-latin characters
        Returns
            str
        """
        string = re.sub(r'[^\x41-\x5a|\x61-\x7a]',r'', string)
        string = re.findall('[A-Z][^A-Z]*', string)
        return ' '.join(string)
    
    for field in ["institution", "major", "degree", "sem"]:
        df[field] = df[field].apply(remove_non_latin)
    
    return df[(df["gpa"].str.isnumeric()==True) & \
            (df["gre_v"].str.isnumeric()==True) & \
            (df["gre_q"].str.isnumeric()==True) & \
            (df["gre_w"].str.isnumeric()==True) & \
            (df["gre_subject"].str.isnumeric()==True)]

def get_ohv(Z_original, Z_, f):
    """
    Get one hot vector representation
    Returns 
        nd.array
    """
    Z_temp = np.array(Z_original)
    Z_ohv = np.zeros((Z_temp[:,f].size, Z_[:,f].max()+1))
    
    Z_ohv[np.arange(Z_temp[:,f].size), Z_temp[:,f]]=1
    return Z_ohv

def naive_bayes(df, isGradCafe=True, k=10):
    """
    NaiveBayes solution
    Returns
        dictionary
    """
    assert isinstance(df, pd.DataFrame), "Not a dataframe!"
    def gradcafe_variables():
        """
        Returns datapoint from pd.DataFrame
        returns 
            Tuple
        """
        fields = df.columns.values.tolist()
        cats = ["Accepted", "Rejected"]
        cat_type = CategoricalDtype(categories=cats, ordered=False)
        X = df[fields[-5:]].to_numpy()
        Y_cat = df[fields[-9]].astype(cat_type)
        Y = Y_cat.cat.codes.to_numpy() + 1
        Y_labels = Y_cat.cat.categories
        Z = list()
        Z_label = list()
        for field in ["institution", "major"]:
            Z_cat = df[field].astype(CategoricalDtype())
            Z.append(Z_cat.cat.codes.to_numpy())
            Z_label.append(Z_cat.cat.categories)
        Z = np.array(Z).T
        return (X, Y, Z, Z_label)
    def kaggle_variables():
        """
        Return datapoints from pd.DataFrame
        returns 
            Tuple
        """
        fields = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']
        X = df[fields].to_numpy()
        Y = (df['Chance of Admit']>0.5).to_numpy()
        Z = list()
        Z_label = list()
        for field in ["Research"]:
            Z_cat = df[field].astype(CategoricalDtype())
            Z.append(Z_cat.cat.codes.to_numpy())
            Z_label.append(Z_cat.cat.categories)
        Z = np.array(Z).T
        return (X, Y, Z, Z_label)
    if isGradCafe:
        X, Y, Z, Z_label = gradcafe_variables()
    else:
        X, Y, Z, Z_label = kaggle_variables()
    kf = KFold(n_splits=k)
    Results = list()
    if isGradCafe:
        f=0
        NB_Classifier = dict()
        min_samples = 10
        for i, field in enumerate(Z_label[f]):
            idx = np.where(Z[:,f] == i)[0]
            if len(idx)>min_samples:
                result = dict()
                accuracy = list()
                y_ = list()
                for i, (train_index, test_index) in enumerate(kf.split(idx)):
                    clf = GaussianNB()
                    X_train = X[idx[train_index]][:,:-2]

                    for k in range(len(Z_label)):
                        if k != f:
                            X_train = np.hstack((X_train, get_ohv(Z[idx[train_index]],Z, k)))
                    NB_Classifier[field] = clf.fit(X_train, Y[idx[train_index]])
                    X_test = X[idx[test_index]][:,:-2]
                    for k in range(len(Z_label)):
                        if k != f:
                            X_test = np.hstack((X_test, get_ohv(Z[idx[test_index]],Z,k)))

                    accuracy.append(clf.score(X_test, Y[idx[test_index]]))
                    y_.append(np.sum(Y[idx[train_index]]==1)/len(train_index))
                result['field'] = field
                result['Mean Accuracy'] = np.mean(accuracy)*100
                result['Max Accuracy'] = np.max(accuracy)*100
                result['Std Dev'] = np.std(accuracy)*100
                result['Number of samples'] = len(idx)
                result['verbose']= "{0}: \n Number of samples: {1} Mean Accuracy: {2:.3f}% Std dev: {3:.3f}% Max Accuracy {4:.3f}% \n".\
                    format(field, len(idx), np.mean(accuracy)*100, np.std(accuracy)*100, np.max(accuracy)*100)
                result['% Lable'] = np.mean(y_)
                Results.append(result)
    else:
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            clf = GaussianNB()
            X_train = X[train_index][:,:-2]
            for k in range(len(Z_label)):
                X_train = np.hstack((X_train, get_ohv(Z[train_index],Z, k)))
            clf.fit(X_train, Y[train_index])
            X_test = X[test_index][:,:-2]
            for k in range(len(Z_label)):
                X_test = np.hstack((X_test, get_ohv(Z[test_index],Z, k)))
            Results.append(clf.score(X_test, Y[test_index]))
    return Results

def main(isGradCafe=True, isKaggle=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    path = "../data/cleaned_data/grad_cafe_admissions_updated.csv"
    if isKaggle:
        path = '../data/cleaned_data/cleaned_kaggle_grad_admissions.csv'
        path = os.path.join(dir_path, path)
        df = import_data(path, False)
        Score = naive_bayes(df, isGradCafe=False, k=10)
        print("Training using Gaussian Naive Bayes(full input vector)")
        print(u"Accuracy after 10-fold cross-validation {0:.3f} \u00B1 {1:.3f}%".format(100*np.mean(Score), 100*np.std(Score)))
        print("Max accuracy {0:.3f}%".format(np.max(Score)*100))
    elif isGradCafe:
        path = '../data/cleaned_data/grad_cafe_admissions_updated.csv'
        path = os.path.join(dir_path, path)
        df = import_data(path)
        df = clean_df(df)
        Results = naive_bayes(df, isGradCafe=True, k=10)
        Results = sorted(Results, key=lambda x : x['Mean Accuracy'], reverse=True)
        print('Top university samples with mean accuracy')
        for i in range(5):
            print(Results[i]['verbose'])
            print("Accepted labels: {0:.2f}%\n".format(100*Results[i]['% Lable']))
    else:
        assert True, "Not implemented!"
    
if __name__ == "__main__":
    # For GradCafe
    main()
    # For Kaggle Dataset
    # isGradCafe=False
    # isKaggle=True
    # main(isGradCafe, isKaggle)