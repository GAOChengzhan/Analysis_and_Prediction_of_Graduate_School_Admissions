# Import Libraries
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import re

if __name__=="__main__":
    # Data is stored in external directory
    path = "../data/cleaned_data/grad_cafe_admissions_updated.csv"
    savePath="../data/cleaned_data/dataGroupByUniversity/"
    fields = []
    rows = []
    # Import data
    # reading csv file
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)# creating a csv reader object
        fields = next(csvreader)# extracting field names through first row
        for row in csvreader:# extracting each data row one by one
            rows.append(row)
    # Convert into dataframe
    df = pd.DataFrame (rows, columns = fields)
    for uni in ['UCLA','Brown University','University Of Pennsylvania']:
        filtered_df = df.loc[df['institution'] == uni]
        filtered_df.to_csv(savePath+ uni+'.csv', index=False)