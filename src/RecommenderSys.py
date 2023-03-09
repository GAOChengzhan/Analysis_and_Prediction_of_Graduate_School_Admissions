# import modules ---------------------------------------------------------------------------------
import pickle
import pathlib
import os
# define the recommenderSys func------------------------------------------------------------------
def recommenderSys(loaded_model,gpa,gre_v,gre_q,gre_w,major,degree):
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
    
    '''
    for i in [gpa,gre_v,gre_q,gre_w]:
        assert isinstance(i,(int,float))
    assert gpa>=0 and gpa <=4.0
    assert gre_q>=0 and gre_q<=170
    assert gre_v>=0 and gre_v<=170
    assert gre_w>=0 and gre_w<=5.0 and gre_w%0.5==0
    assert isinstance(major,str) and isinstance(degree,str)
    recommendations=[]
    return recommendations


if __name__=="__main__":
    # load model----------------------------------------------------------------------------------
    
    filename=r"./model/best_model.model"
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
    except FileNotFoundError:
        print("Cannot find the model file in the directory:{}".format(filename))
    
    HEADER="According to the recommendation system, the following listed universities are for you\
        to consider:"
    # input---------------------------------------------------------------------------------------
    gpa = input("GPA(out of 4.0):")
    gre_v = input("GRE verbal score:")
    gre_q = input("GRE quantitative score:")
    gre_w = input("GRE writing score:")
    major = input("Major:")
    degree = input("Degree:")

    # input validation----------------------------------------------------------------------------
    try:
        gpa=float(gpa)
        gre_v=float(gre_v)
        gre_q=float(gre_q)
        gre_w=float(gre_w)
    except:
        print("Please type in valid numbers!")
    assert gpa>=0 and gpa <=4.0
    assert gre_q>=0 and gre_q<=170
    assert gre_v>=0 and gre_v<=170
    assert gre_w>=0 and gre_w<=5.0 and gre_w%0.5==0

    # run recommendation function ---------------------------------------------------------------
    recommendedUni=recommenderSys(loaded_model,gpa,gre_v,gre_q,gre_w,major,degree)
    print(HEADER)

    # list the results --------------------------------------------------------------------------
    for i,uni in enumerate(recommendedUni):
        print("{}. {}".format(i+1,uni))
