from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import matplotlib.pyplot as plt

import matplotlib as mpl
from implicit import bpr
import seaborn as sns
import pandas as pd
import numpy as np

if __name__=="__main__":
    HEAD = ["submissionDate","institution","major","degree","notif_result","notif_date","gpa","gre","gre_w"]
    data = pd.read_csv('../data/cleaned_data/cleaned_grad_cafe_admissions.csv',)

    # replace ['E-mail','Wait listed','Other'] with "Rejected"
    datafiltered=data.replace(to_replace=['E-mail','Wait listed','Other'],value='Rejected')
    # filter data with unnormal results
    datafiltered=datafiltered.loc[data['notif_result'].isin(['Accepted','Rejected'])] 
    # filter data with unnormal gpa
    datafiltered=datafiltered[datafiltered["gpa"].map(lambda x:x>0 and x<4.0)]
    # filter data with unnormal gre
    datafiltered=datafiltered[datafiltered["gre"].map(lambda x:x>300 and x<=340)]
    datafiltered=datafiltered[datafiltered["gre_w"].map(lambda x:x>0.0 and x<=5.0)]
    # data length
    print("data only contains accepted and rejected:{}".format(len(datafiltered)))
    # Split the data into training and testing sets
    X = datafiltered.iloc[:,[1,2,3,4,6,7,8,9]]  # select all columns except the last one
    y = datafiltered.iloc[:, 5]   # select the last column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

    university_rank=pd.read_csv('../data/university_rank/university_rank.csv',names=['rank','Uni_Name','Country',\
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    dict_university_rank=defaultdict(list)
    university_set=set()
    university_name=[]
    university_value=[]
    for i in range(len(university_rank)):
        uni_name=str(university_rank.iloc[i,1])
        if i<50:
            uni_rank=int(university_rank.iloc[i,0])//10
        elif i<100:
            uni_rank=5
        elif i<200:
            uni_rank=6
        elif i<500:
            uni_rank=7
        else:
            uni_rank=8
        dict_university_rank[uni_name]=uni_rank
        if uni_name not in university_set:
            university_set.add(uni_name)
            university_name.append(uni_name)
            university_value.append(uni_rank)
    data_UniRank = datafiltered[datafiltered["institution"].map(lambda x: x in university_name)]
    data_UniRank = data_UniRank.replace(to_replace=university_name,value=university_value)
    print("Length of data which are interacted with university rank is:{}".format(len(data_UniRank)))