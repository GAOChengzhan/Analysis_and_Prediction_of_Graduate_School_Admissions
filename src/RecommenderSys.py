# import modules ---------------------------------------------------------------------------------
import pickle
import pandas as pd
import torch
from ANN_class import ANN
import pathlib
import os
# define map uni to rank func--------------------------------------------------------------------
def rankMap(rank):
    '''
    This function aims to map each university to its predefined class according to its rank.
    *Input: rank, the rank of university
    *Output: the map code
    '''
    i=rank
    university_rank=pd.read_csv('./data/university_rank/university_rank.csv',names=\
        ['rank','Uni_Name','Country','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
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
    return uni_rank
# define the recommenderSys func------------------------------------------------------------------
def recommenderSys(gpa,gre_v,gre_q,gre_w,major,degree,N=10,mode=0):
    '''
    This function aims to take in the input information of the students and uses the trained model 
    to list the universities with high admission probability as recommendation universities, which
    servers as reference for students when application. 
    *Input:
        gpa: overall grade point average
        gre_v: score of gre verbal part
        gre_q: score of gre quantitative part
        gre_w: score of gre writing part
        major: the major of the student
        degree: the degree of the student want to pursue
    *Output:
        recommendations: a list of the name of recommended universities
    *Modules:
        torch
    *Dependencies:
        ANN_cafe_Prediction_Pytorch.ipynb
    '''
    for i in [gpa,gre_v,gre_q,gre_w]:
        assert isinstance(i,(int,float))
    assert gpa>=0 and gpa <=4.0
    assert gre_q>=0 and gre_q<=170
    assert gre_v>=0 and gre_v<=170
    assert gre_w>=0 and gre_w<=5.0 and gre_w%0.5==0
    assert isinstance(major,str) and isinstance(degree,str)
    cnt=0
    recommendations=[]

    # load universities---------------------------------------------------------------------------
    university_rank=pd.read_csv('./data/university_rank/university_rank.csv',names=\
        ['rank','Uni_Name','Country','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    uni_name=[]
    for i in range(len(university_rank)):
        uni_name.append(university_rank.iloc[i][1])

    if mode==0:
        Threshold=0.5
        for i,insititution in enumerate(uni_name[:20]):
            uni_rank=rankMap(i)
            # Define the input tensor
            x = torch.tensor([uni_rank, gpa, gre_q+gre_v], dtype=torch.float32)

            # Load the trained model
            model = ANN(input_dim=3, output_dim=1)
            model.load_state_dict(torch.load('./model/saved_model.pt'))

            # Make a prediction for the input x
            model.eval()
            with torch.no_grad():
                y_pred = model(x)
                y_pred = y_pred.item()
            print(y_pred)
            if(y_pred>=Threshold):
                recommendations.append(insititution)
                cnt+=1
            if(cnt>=N):
                break
    elif mode==1:
        pass   

    return recommendations


if __name__=="__main__":


    # load model----------------------------------------------------------------------------------
    filename=r"./model/best_model.model"
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
    except FileNotFoundError:
        print("Cannot find the model file in the directory:{}".format(filename))
    
    HEADER="According to the recommendation system,"+\
            " the following listed universities are for you to consider:"
    # # input---------------------------------------------------------------------------------------
    # gpa = input("GPA(out of 4.0):")
    # gre_v = input("GRE verbal score:")
    # gre_q = input("GRE quantitative score:")
    # gre_w = input("GRE writing score:")
    # major = input("Major:")
    # degree = input("Degree:")

    # # input validation----------------------------------------------------------------------------
    # try:
    #     gpa=float(gpa)
    #     gre_v=float(gre_v)
    #     gre_q=float(gre_q)
    #     gre_w=float(gre_w)
    # except:
    #     print("Please type in valid numbers!")
    # assert gpa>=0 and gpa <=4.0
    # assert gre_q>=0 and gre_q<=170
    # assert gre_v>=0 and gre_v<=170
    # assert gre_w>=0 and gre_w<=5.0 and gre_w%0.5==0

    # run recommendation function ---------------------------------------------------------------
    recommendedUni=recommenderSys(gpa=4.0,gre_v=165,gre_q=165,gre_w=4.0,\
                                  major="Computer Science",degree="Masters")
    print(HEADER)

    # list the results --------------------------------------------------------------------------
    for i,uni in enumerate(recommendedUni):
        print("{}. {}".format(i+1,uni))
