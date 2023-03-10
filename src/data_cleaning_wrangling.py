import numpy as np
import pandas as pd
import os

class Clean():

    def __init__(self, kaggle_path, grad_cafe_path):
        # Assertions check
        assert isinstance(kaggle_path, str) and os.path.exists(kaggle_path)
        assert isinstance(grad_cafe_path, str) and os.path.exists(kaggle_path)

        self.__kaggle_path = kaggle_path
        self.__grad_cafe_path = grad_cafe_path

    def clean_kaggle_dataset(self, export=False):
        '''
        Clean and wrangle the raw kaggle graduate admission dataset
        relative to its path

        :param export: select True to export the cleaned dataset, else False
        :return: cleaned kaggle dataset pandas dataframe
        '''

        assert isinstance(export, bool)

        # Create dataframe
        kaggle = pd.read_csv(self.__kaggle_path, low_memory=False)
        kaggle = kaggle.rename(str.strip, axis='columns')

        # Drop uninterested columns and drop null containing rows
        kaggle = kaggle.drop(columns=['Serial No.'])
        kaggle = kaggle.dropna().reset_index(drop=True)

        # Scale GPA to 4.0 scale
        kaggle.iloc[:, 5] = kaggle.iloc[:, 5].apply(lambda x: (x / 10.0) * 4.0)

        # Filter dataset for only valid GRE, TOEFL, University Ranking, SOP, LOR
        # CGPA, and Chance of Admission ranges
        kaggle = kaggle[kaggle['GRE Score'].between(0, 340)]
        kaggle = kaggle[kaggle['TOEFL Score'].between(0, 120)]
        kaggle = kaggle[kaggle['University Rating'].between(0, 5)]
        kaggle = kaggle[kaggle['SOP'].between(0, 5)]
        kaggle = kaggle[kaggle['LOR'].between(0, 5)]
        kaggle = kaggle[kaggle['CGPA'].between(0, 4)]
        kaggle = kaggle[kaggle['Chance of Admit'].between(0, 1)]

        if export:
            sub_directory = os.path.dirname(self.__grad_cafe_path)
            data_directory = os.path.dirname(sub_directory)
            kaggle.to_csv(data_directory + '/cleaned_data/cleaned_kaggle_grad_admissions.csv')
        
        return kaggle

    def clean_grad_cafe_dataset(self, export=False):
        '''
        Clean and wrangle the raw GradCafe graduate admission dataset
        relative to its path

        :param export: select True to export the cleaned dataset, else False
        :return: cleaned GradCafe dataset pandas dataframe
        '''

        assert isinstance(export, bool)

        # Create dataframe
        grad_cafe = pd.read_csv(self.__grad_cafe_path, sep=';', low_memory=False)
        grad_cafe = grad_cafe.rename(str.strip, axis='columns')

        # Drop uninterested columns and remove null containing rows
        grad_cafe = grad_cafe.drop(columns=['submissionId', 
                                            'sem', 
                                            'notif_method',
                                            'studentType',
                                            'gre_subject',
                                            'notes',
                                            'gre_w'])

        grad_cafe = grad_cafe.dropna().reset_index(drop=True)

        # Create a new cumulative GRE column
        gre_cumulative = grad_cafe.loc[:, 'gre_v'].add(grad_cafe.loc[:,'gre_q'])
        grad_cafe.insert(7, 'gre', gre_cumulative)

        # Drop GRE Verbal and Quantitative columns
        grad_cafe = grad_cafe.drop(columns=['gre_v', 'gre_q'])

        # Filter dataset for only valid GRE and GPA scores 
        grad_cafe = grad_cafe[(0 <= grad_cafe['gpa']) & (grad_cafe['gpa'] <= 4)]
        grad_cafe = grad_cafe[(0 <= grad_cafe['gre']) & (grad_cafe['gre'] <= 340)]

        if export:
            sub_directory = os.path.dirname(self.__grad_cafe_path)
            data_directory = os.path.dirname(sub_directory)
            grad_cafe.to_csv(data_directory + '/cleaned_data/cleaned_grad_cafe_admissions.csv') 
        
        return grad_cafe

    def unlabled_grad_cafe_dataset(self, export=False):
        '''
        Export the raw unlabled GradCafe graduate admission dataset
        relative to its path

        :param export: select True to export the unlabled dataset, else False
        :return: unlabled GradCafe dataset pandas dataframe
        '''

        assert isinstance(export, bool)

        # Create dataframe
        unlabled = pd.read_csv(self.__grad_cafe_path, sep=';', low_memory=False)
        unlabled = unlabled.rename(str.strip, axis='columns')

        # Drop uninterested columns and remove null containing rows
        unlabled = unlabled.drop(columns=['submissionId', 
                                        'sem', 
                                        'notif_method',
                                        'studentType',
                                        'gre_subject',
                                        'notes'])

        # Filter for rows with NaN
        unlabled = unlabled[unlabled.isna().any(axis=1)]
        
        # Get rows with both verbal and quantitative scores
        gre_complete = unlabled[unlabled[['gre_v', 'gre_q']].notnull().all(axis=1)]

        # Create a new cumulative GRE column
        gre_cumulative = gre_complete.loc[:, 'gre_v'].add(gre_complete.loc[:,'gre_q'])
        gre_complete.insert(7, "gre", gre_cumulative)

        # Drop GRE Verbal and Quantitative columns
        gre_complete = gre_complete.drop(columns=['gre_v', 'gre_q'])

        # Get rows missing verbal and/or quantitative scores
        gre_incomplete = unlabled[~unlabled[['gre_v', 'gre_q']].notnull().all(axis=1)]

        # Create a NaN GRE column
        gre_incomplete.insert(7, "gre", np.nan)

        # Drop GRE Verbal and Quantitative columns
        gre_incomplete = gre_incomplete.drop(columns=['gre_v', 'gre_q'])

        # Combine complete and incomplete unlabled dataframes
        unlabled = pd.concat([gre_complete, gre_incomplete]).reset_index()

        if export:
            sub_directory = os.path.dirname(self.__grad_cafe_path)
            data_directory = os.path.dirname(sub_directory)
            unlabled.to_csv(data_directory + '/cleaned_data/unlabled_grad_cafe_admissions.csv')

        return unlabled


# Main start here, for testing
if __name__ == '__main__':
    kaggle_path = 'data/raw_data/kaggle_grad_admissions.csv'
    grad_cafe_path = 'data/raw_data/submissions.csv'

    clean = Clean(kaggle_path, grad_cafe_path)
    clean.clean_kaggle_dataset(True)
    clean.clean_grad_cafe_dataset(True)
    clean.unlabled_grad_cafe_dataset(True)