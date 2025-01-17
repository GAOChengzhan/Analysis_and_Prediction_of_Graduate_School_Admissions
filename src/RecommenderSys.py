# import modules ---------------------------------------------------------------------------------
import pickle
import pandas as pd
import torch
from ANN_class import ANN
import pathlib
import os
from collections import defaultdict

# define ThresholdAdapter func--------------------------------------------------------------------
def ThresholdAdapter(Threshold,rank,mode):
    if mode==0 or mode==1:
        NewTH=Threshold-(5-rank)*0.08
    
    return NewTH
# define map uni to rank func--------------------------------------------------------------------
def rankMap(rank,mode):
    '''
    This function aims to map each university to its predefined class according to its rank.
    *Input: rank, the rank of university
    *Output: the map code
    '''
    i=rank
    university_rank=pd.read_csv('../data/university_rank/university_rank.csv',names=\
        ['rank','Uni_Name','Country','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    if mode==0:
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
    elif mode==1:
        if i<=20:
            uni_rank=(int(university_rank.iloc[i,0])-1)//10+1
        elif i<=50:
            uni_rank=3
        elif i<=200:
            uni_rank=4
        else:
            uni_rank=5

    return uni_rank
# define the recommenderSys func------------------------------------------------------------------
def recommenderSys(gpa,gre_v,gre_q,gre_w,TOEFL,major,degree,SOP,LOR,Research,N=15,mode=0):
    '''
    This function aims to take in the input information of the students and uses the trained model 
    to list the universities with high admission probability as recommendation universities, which
    servers as reference for students when application. 
    *Input:
        gpa: overall grade point average
        gre_v: score of gre verbal part
        gre_q: score of gre quantitative part
        gre_w: score of gre writing part
        TOEFL: the TOEFL score
        major: the major of the student
        degree: the degree of the student want to pursue 
        SOP: Statement of Purpose Strength ( out of 5 )
        LOR: Letter of Recommendation Strength ( out of 5 )
        Research: Research Experience ( either 0 or 1 )

        N: the number of universities the user wants to get recommended
        mode: select one from two kinds of model to do the recommendation
    *Output:
        recommendations: a list of the name of recommended universities
    *Modules:
        1. torch
        2. pickle
        3. pandas
        4. ANN_class
        5. collections
    *Dependencies:
        ANN_cafe_Prediction_Pytorch.ipynb
    '''
    for i in [gpa,gre_v,gre_q,gre_w,TOEFL,SOP,LOR,Research]:
        try:
            assert isinstance(i,(int,float))
            assert i>=0
        except AssertionError:
            print(i)
    assert gpa <=4.0
    assert gre_q<=170 and gre_v<=170 and gre_w<=5.0 and gre_w%0.5==0
    assert TOEFL<=120 and SOP<=5 and LOR<=5 and Research<2
    assert isinstance(major,str) and isinstance(degree,str)
    assert isinstance(mode,int) and mode>=0 and mode<3
    cnt=0
    recommendations=[]
    chances=[]
    rankloc=defaultdict(int)
    # load universities-----------------------------------------------------------------------------
    university_rank=pd.read_csv('../data/university_rank/university_rank.csv',names=\
        ['rank','Uni_Name','Country','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    uni_name=[]
    for i in range(len(university_rank)):
        uni_name.append(university_rank.iloc[i][1])
    # mode 0----------------------------------------------------------------------------------------
    if mode==0:
        Threshold=0.5
        # Load the trained model
        model = ANN(input_dim=3, output_dim=1)
        try:
            model.load_state_dict(torch.load('../model/saved_model.pt'))
        except FileNotFoundError:
            print("Cannot find the model file in the directory")
        model.eval()
    # mode 1-----------------------------------------------------------------------------------------
    elif mode==1:
        Threshold=0.8
        try:
            with open('../model/LinearReg.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
        except FileNotFoundError:
            print("Cannot find the model file in the directory")
    # mode 2-----------------------------------------------------------------------------------------
    elif mode==2:
        pass

    while Threshold>0.2:
        flag=False
        for i,insititution in enumerate(uni_name):
            uni_rank=rankMap(i,mode)
            # mode0----------------------------------------------------------------------------------
            if mode==0:
                x = torch.tensor([uni_rank, gpa, gre_q+gre_v], dtype=torch.float32)
                with torch.no_grad():
                    y_pred = model(x)
                    y_pred = y_pred.item()
            # mode1----------------------------------------------------------------------------------
            elif mode==1:
                names = ['GRE Score', 'TOEFL Score','University Rating', 'SOP', \
                        'LOR', 'CGPA', 'Research',]
                x=pd.DataFrame([[gre_q+gre_v,TOEFL,uni_rank,SOP,LOR,gpa,Research]],columns=names)
                y_pred,=loaded_model.predict(x)
                Threshold=ThresholdAdapter(Threshold,uni_rank,mode)
            # mode 2-----------------------------------------------------------------------------------------
            elif mode==2:
                pass
            
            if (y_pred>Threshold):
                rankloc[uni_rank]=rankloc.get(uni_rank,0)+1
                if rankloc[uni_rank]>3:
                    continue
                recommendations.append(insititution)
            
                chances.append(y_pred)
                cnt+=1
            if(cnt>=N):
                flag=True
                break
        if flag:
            break
        Threshold-=0.1

    return recommendations,chances


if __name__=="__main__":
    max_len = 59
    HEADER="According to the recommendation system,"+\
            " the following listed universities are for you to consider:"
    NameChance="------------------------Name of the Uni-----------------------Chance-------"
    # input---------------------------------------------------------------------------------------
    gpa = input("GPA(out of 4.0): ")
    gre_v = input("GRE verbal score: ")
    gre_q = input("GRE quantitative score: ")
    # gre_w = input("GRE writing score:")
    # major = input("Major:")
    # degree = input("Degree:")
    TOEFL=input("TOEFL score: ")
    SOP = input("Statement of Purpose Strength ( out of 5 ): ")
    LOR = input("Letter of Recommendation Strength ( out of 5 ): ")
    Research = input("Research Experience ( either 0 or 1 ): ")

    # input validation----------------------------------------------------------------------------
    try:
        gpa=float(gpa)
        gre_v=float(gre_v)
        gre_q=float(gre_q)
        TOEFL=float(TOEFL)
        SOP=int(SOP)
        LOR=int(LOR)
        Research=int(Research)
        # gre_w=float(gre_w)
    except:
        print("Please type in valid numbers!")

    # assert gre_w>=0 and gre_w<=5.0 and gre_w%0.5==0


    # run recommendation function ---------------------------------------------------------------
    recommendedUni,chances=recommenderSys(gpa=gpa,gre_v=gre_v,gre_q=gre_q,gre_w=4.0,TOEFL=TOEFL,\
                                  major="Computer Science",degree="Masters",\
                                  SOP=SOP,LOR=LOR,Research=Research,\
                                  mode=1)
    
    # list the results --------------------------------------------------------------------------
    print(HEADER)
    print(NameChance)
    for i,uni in enumerate(recommendedUni):
        print("{:02d}.{:<{}}{:.2f}%".format(i+1,uni, max_len,chances[i]*100).replace(" ", "."))