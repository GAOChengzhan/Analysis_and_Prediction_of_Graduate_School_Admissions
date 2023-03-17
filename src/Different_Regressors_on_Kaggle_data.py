
# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

# Load the data into a Pandas dataframe
names = ['GRE Score', 'TOEFL Score','University Rating', 'SOP', 
         'LOR', 'CGPA', 'Research', 'Chance of Admit ']
data = pd.read_csv('../data/cleaned_data/cleaned_kaggle_grad_admissions.csv',)

def visualization_for_predict_data(data_observed, data_predicted, label_observed, label_predicted):
    mpl.rcParams['figure.dpi']=80
    plt.rcParams['figure.figsize'] = [5,5]
    plt.scatter(data_observed, data_predicted)
    max_v = np.max([data_predicted] + [data_observed]) * 1.1
    min_v = np.min([data_predicted] + [data_observed]) * 0.9
    plt.plot([min_v,max_v], [min_v,max_v], "r-")
    plt.xlabel(label_observed)
    plt.ylabel(label_predicted)

def MSE_cal(obs,pred):
    assert len(obs)==len(pred)
    sumErr=0
    for i in range(len(pred)):
        squareErr=((pred[i]-obs.iloc[i])**2)
        sumErr+=squareErr
    return sumErr/len(pred)

def Accuracy_cal(obs,pred):
    assert len(obs)==len(pred)
    cnt=0
    if type(y_test)==type(np.array([])):
        for i in range(len(pred)):
            if pred[i]>0.5 and obs[i]>0.5:
                cnt+=1
            elif pred[i]<0.5 and obs[i]<0.5:
                cnt+=1
        return cnt/len(pred)
    else:
        for i in range(len(pred)):
            if pred[i]>0.5 and obs.iloc[i]>0.5:
                cnt+=1
            elif pred[i]<0.5 and obs.iloc[i]<0.5:
                cnt+=1
        return cnt/len(pred)
    
if __name__=="__main__":
    # Split the data into training and testing sets
    X = data.iloc[:, -8:-1]  # select all columns except the last one
    y = data.iloc[:, -1]   # select the last column
    yAcc= y.apply(lambda x: 1 if x>0.5 else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Linear Regression (by Christoffel)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred=reg.predict(X_test)
    regSave = LinearRegression().fit(X, y)
    with open('../model/LinearReg.pkl', 'wb') as file:
        pickle.dump(regSave, file)

    ### visualization
    visualization_for_predict_data(y_test, y_pred, "y_train", "y_pred")
    ### MSE calculation
    print("MSE calculation")
    MSE = MSE_cal(y_test,y_pred)
    print(MSE)
    print("reg.coef_:{};".format(reg.coef_))
    ### Accuracy calculation
    print("Accuracy calculation")
    accuracy=Accuracy_cal(y_test,y_pred)
    print(accuracy)

    ### only pick the gre and gpa features
    print("only pick the gre and gpa features")
    X_trainG=X_train.iloc[:,[1,2,6]]
    X_testG=X_test.iloc[:,[1,2,6]]
    reg = LinearRegression().fit(X_trainG, y_train)
    y_predG=reg.predict(X_testG)
    MSE = MSE_cal(y_test,y_predG)
    print(MSE)
    # - If only pick the three features with highest correlation, MSE is worse than using all feartures
    # - This indicates that other features are still helpful for the prediction.

    # ## Other models(by Chritoffel)
    # - DecisionTreeRegressor
    # - RandomForestRegressor
    # - XGBRegressor
    # - KNeighborsRegressor

    # ### Standard scalar
    scaler = StandardScaler()
    dataArray=np.array(data)
    Xdata=dataArray[:, :-1]
    scaler.fit(Xdata)
    standard_data = scaler.transform(Xdata)
    # Split the data into training and testing sets
    X = standard_data  # select all columns except the last one
    y =dataArray[:, -1]   # select the last column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ### Use different regressor models
    print("Use different regressor models")
    classifiers = [
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        XGBRegressor(),
        KNeighborsRegressor(),
        ]
    classiferName=['DecisionTreeRegressor','RandomForestRegressor','XGBRegressor','KNeighborsRegressor']

    for i,classifier in enumerate(classifiers):
        pipe = Pipeline(steps=[('classifier', classifier)])
        pipe.fit(X_train, y_train)
        y_pred=pipe.predict(X_test)
        MSE = MSE_cal(y_test,y_pred)
        Acc = Accuracy_cal(y_test,y_pred)
        print("{} MSE: {} Acc: {}" .format(classiferName[i],MSE,Acc))
