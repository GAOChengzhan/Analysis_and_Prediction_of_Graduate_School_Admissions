# Import Libraries
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import re
# Data is stored in external directory
path = "../data/cleaned_data/grad_cafe_admissions_updated.csv"
savePath="../data/cleaned_data/dataGroupByUniversity/"
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
for uni in ['UCLA','Brown University','University Of Pennsylvania']:
    filtered_df = df.loc[df['institution'] == uni]
    filtered_df.to_csv(savePath+ uni+'.csv', index=False)