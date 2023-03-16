import numpy as np
import pandas as pd
import os
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

class KaggleAnalysis():

    def __init__(self, kaggle_path):
        # Assertions check
        assert isinstance(kaggle_path, str) and os.path.exists(kaggle_path)

        self.kaggle_df = pd.read_csv(kaggle_path, low_memory=False)
        self.kaggle_df = self.kaggle_df.drop(columns='Unnamed: 0')

    def generate_statistics(self):
        '''
        Helper for generating pandas statistics
        '''
        return self.kaggle_df.describe()

    def generate_boxplots(self, export=False):
        '''
        Generate box plot summary of the kaggle dataset

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.5)
        fig.delaxes(axes[1][3])

        # GRE Scores
        sns.boxplot(y=self.kaggle_df['GRE Score'], ax=axes[0][0], color='red')

        # TOEFL Scores
        sns.boxplot(y=self.kaggle_df['TOEFL Score'], ax=axes[0][1], color='orange')

        # University Rating
        sns.boxplot(y=self.kaggle_df['University Rating'], ax=axes[0][2], color='yellow')

        # Statement of Purposes
        sns.boxplot(y=self.kaggle_df['SOP'], ax=axes[0][3], color='green')

        # Letter of Recommendations
        sns.boxplot(y=self.kaggle_df['LOR'], ax=axes[1][0], color='blue')

        # CGPA
        sns.boxplot(y=self.kaggle_df['CGPA'], ax=axes[1][1], color='indigo')

        # Chance of Admission
        sns.boxplot(y=self.kaggle_df['Chance of Admit'], ax=axes[1][2], color='violet')

        if export:
            plt.savefig('analysis_outputs/kaggle_boxplots.png')

        plt.show()
        
    def generate_distribution_plots(self, export=False):
        '''
        Generate distribution summary of the kaggle dataset

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        # GRE Scores
        sns.histplot(data=self.kaggle_df, x='GRE Score', ax=axes[0][0], linewidth=0.3, edgecolor='white', color='#738bd9')

        # TOEFL Scores
        sns.histplot(data=self.kaggle_df, x='TOEFL Score', ax=axes[0][1], linewidth=0.3, edgecolor='white', color='#9bb5f0')

        # University Rating
        sns.histplot(data=self.kaggle_df, x='University Rating', linewidth=0.3, edgecolor='white', ax=axes[0][2], color='#c1d2f1')

        # Statement of Purposes
        sns.histplot(data=self.kaggle_df, x='SOP', ax=axes[0][3], linewidth=0.3, edgecolor='white', color='#dddcdc')

        # Letter of Recommendations
        sns.histplot(data=self.kaggle_df, x='LOR', ax=axes[1][0], linewidth=0.3, edgecolor='white', color='#ecc7b6')

        # CGPA
        sns.histplot(data=self.kaggle_df, x='CGPA', ax=axes[1][1], linewidth=0.3, edgecolor='white', color='#e5a089')

        # Research
        sns.histplot(data=self.kaggle_df, x='CGPA', ax=axes[1][2], linewidth=0.3, edgecolor='white', color='#cc6c5e')

        # Chance of Admission
        sns.histplot(data=self.kaggle_df, x='Chance of Admit', ax=axes[1][3], linewidth=0.3, edgecolor='white', color='#8f4c40')

        if export:
            plt.savefig('analysis_outputs/kaggle_distribution_plots.png')

        plt.show()

    def generate_heat_map(self, export):
        '''
        Generate heatmap summary of the kaggle dataset labels

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        plt.figure(figsize=(15, 8))
        axes = plt.axes()

        # Generate a mask for the upper triangle
        corr = self.kaggle_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap='coolwarm', ax=axes, annot=True, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'color':'w'})
        axes.set_title('Kaggle Dataset Labels Correlation Matrix')

        if export:
            plt.savefig('analysis_outputs/kaggle_heatmap.png')

        plt.show()

    def generate_correlation_plots(self, export):
        '''
        Generate correlation plots of the kaggle dataset labels for
        (1) GRE Score vs Chance of Admission
        (2) CGPA vs Chance of Admission
        (3) CGPA vs GRE Score

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        # Scatter plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # GRE Score vs Chance of Admission
        sns.regplot(x='GRE Score', y='Chance of Admit', data=self.kaggle_df, fit_reg=True, ax=axes[0])

        # CGPA vs Chance of Admission
        sns.regplot(x='CGPA', y='Chance of Admit', data=self.kaggle_df, fit_reg=True, ax=axes[1])

        # CGPA vs GRE Score
        sns.regplot(x='CGPA', y='GRE Score', data=self.kaggle_df, fit_reg=True, ax=axes[2])

        if export:
            plt.savefig('analysis_outputs/kaggle_correlation_plots.png')

        plt.show()

    def generate_correlation_coefficients(self):
        '''
        Generate correlation coefficients of the kaggle dataset labels for
        (1) GRE Score vs Chance of Admission
        (2) CGPA vs Chance of Admission
        (3) CGPA vs GRE Score

        :return: dictionary of correlation coefficients
        '''

        # GRE Score vs Chance of Admission
        gre_r, _ = scipy.stats.pearsonr(x=self.kaggle_df['GRE Score'], y=self.kaggle_df['Chance of Admit'])

        # CGPA vs Chance of Admission
        cgpa_r, _ = scipy.stats.pearsonr(x=self.kaggle_df['CGPA'], y=self.kaggle_df['Chance of Admit'])

        # CGPA vs GRE Score
        cgpa_gre_r, _ = scipy.stats.pearsonr(x=self.kaggle_df['CGPA'], y=self.kaggle_df['GRE Score'])

        return {'gre_chance': gre_r, 'cgpa_chance': cgpa_r, 'cgpa_gre': cgpa_gre_r}

        
# Main start here, for testing
if __name__ == '__main__':
    kaggle_path = 'data/cleaned_data/cleaned_kaggle_grad_admissions.csv'

    analysis = KaggleAnalysis(kaggle_path)
    analysis.generate_heat_map(True)