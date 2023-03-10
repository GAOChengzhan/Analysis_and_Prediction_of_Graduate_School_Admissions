import numpy as np
import pandas as pd
import os
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

class GradCafeAnalysis():

    def __init__(self, grad_cafe_path):
        # Assertions check
        assert isinstance(grad_cafe_path, str) and os.path.exists(grad_cafe_path)

        self.grad_cafe_df = pd.read_csv(grad_cafe_path, low_memory=False)
        self.grad_cafe_df = self.grad_cafe_df.drop(columns='Unnamed: 0')

    def generate_statistics(self):
        '''
        Helper for generating pandas statistics
        '''
        return self.grad_cafe_df.describe()

    def generate_boxplots(self, export=False):
        '''
        Generate box plot summary of the GradCafe dataset

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        # GPA 
        sns.boxplot(y=self.grad_cafe_df['gpa'], ax=axes[0], color='red')

        # GRE Scores
        sns.boxplot(y=self.grad_cafe_df['gre'], ax=axes[1], color='orange')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_boxplots.png')

        plt.show()

    def generate_distribution_plots(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        self.grad_cafe_df.hist(figsize=(20,5), layout=(1,2))

        if export:
            plt.savefig('analysis_outputs/grad_cafe_distribution_plots.png')

        plt.show()

    def generate_heat_map(self, export=False):
        '''
        Generate heatmap summary of the GradCafe dataset labels

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        axes = plt.axes()

        grad_cafe_correlation_matrix = self.grad_cafe_df[['gpa', 'gre']].corr()
        grad_cafe_correlation_matrix = sns.heatmap(grad_cafe_correlation_matrix, ax=axes, annot=True)
        axes.set_title('GradCafe Dataset Labels Correlation Matrix')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_headmap.png')

        plt.show()

    def generate_correlation_plot(self, export=False):
        '''
        Generate correlation plots of the GradCafe dataset labels for
        (1) GPA vs GRE Score

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        sns.regplot(x='gpa', y='gre', data=self.grad_cafe_df, fit_reg=True)

        if export:
            plt.savefig('analysis_outputs/grad_cafe_correlation_plot.png')

        plt.show()

    def generate_correlation_coefficients(self):
        '''
        Generate correlation coefficients of the kaggle dataset labels for
        (1) GPA vs GRE Score

        :param export: select True to export the boxplot pictures, else False
        '''

        gre_gpa_r, _ = scipy.stats.pearsonr(x=self.grad_cafe_df['gpa'], y=self.grad_cafe_df['gre'])

        return {'gpa_gre': gre_gpa_r}

    def generate_application_result_chart(self, export=False):
        '''
        Generate pie chart summary of the GradCafe dataset application results

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        result_counts = self.grad_cafe_df['notif_result'].value_counts()
        result_counts.plot.pie()

        if export:
            plt.savefig('analysis_outputs/grad_cafe_application_result_chart.png')

        plt.show()

    def generate_averages_by_application_results(self, export=False):
        '''
        Generate the plot on the average gpa and gre for each application result
        '''

        result_by = self.grad_cafe_df.groupby('notif_result')[['gpa', 'gre']].agg('mean')
        result_by = result_by.reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Distribution of gpa by result
        sns.barplot(x='notif_result', y='gpa', data=result_by, ax=axes[0]).set_yscale("log")

        # Distribution of gre by result
        sns.barplot(x='notif_result', y='gre', data=result_by, ax=axes[1]).set_yscale("log")

        if export:
            plt.savefig('analysis_outputs/grad_cafe_averages_by_application_result.png')

        plt.show()


class ByMajor(GradCafeAnalysis):

    def __init__(self, grad_cafe_path):
        super().__init__(grad_cafe_path)

        self.engineering_df = self.grad_cafe_df[self.grad_cafe_df['major'].str.contains('engineering|Engineering')]
        self.sciences_df = self.grad_cafe_df[self.grad_cafe_df['major'].str.contains('science|Science')]

    def generate_engineering_statistics(self):
        '''
        Helper for generating pandas statistics for engineering subset of majors
        '''
        return self.engineering_df.describe()
    
    def generate_sciences_statistics(self):
        '''
        Helper for generating pandas statistics for sciences subset of majors
        '''
        return self.sciences_df.describe()
    
    def generate_distribution_plots_engineering(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset by engineering major

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        self.engineering_df.hist(figsize=(20,5), layout=(1,2))

        if export:
            plt.savefig('analysis_outputs/grad_cafe_engineering_distribution_plots.png')

        plt.show()

    def generate_distribution_plots_sciences(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset by sciences major

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        self.sciences_df.hist(figsize=(20,5), layout=(1,2))

        if export:
            plt.savefig('analysis_outputs/grad_cafe_sciences_distribution_plots.png')

        plt.show()

    def generate_distribution_by_result_engineering(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset by engineering major for each result

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        engineering_result_by = self.engineering_df.groupby('notif_result')[['gpa', 'gre']].agg('mean')
        engineering_result_by = engineering_result_by.reset_index()

        # Grouby distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Distribution of gpa by result
        sns.barplot(x='notif_result', y='gpa', data=engineering_result_by, ax=axes[0]).set_yscale("log")

        # Distribution of gre by result
        sns.barplot(x='notif_result', y='gre', data=engineering_result_by, ax=axes[1]).set_yscale("log")

        if export:
            plt.savefig('analysis_outputs/grad_cafe_engineering_distribution_by_result.png')

        plt.show()

    def generate_distribution_by_result_sciences(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset by sciences major for each result

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        sciences_result_by = self.sciences_df.groupby('notif_result')[['gpa', 'gre']].agg('mean')
        sciences_result_by = sciences_result_by.reset_index()

        # Grouby distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Distribution of gpa by result
        sns.barplot(x='notif_result', y='gpa', data=sciences_result_by, ax=axes[0]).set_yscale("log")

        # Distribution of gre by result
        sns.barplot(x='notif_result', y='gre', data=sciences_result_by, ax=axes[1]).set_yscale("log")

        if export:
            plt.savefig('analysis_outputs/grad_cafe_sciences_distribution_by_result.png')

        plt.show()


# Main start here, for testing
if __name__ == '__main__':
    grad_cafe_path = 'data/cleaned_data/cleaned_grad_cafe_admissions.csv'

    analysis = ByMajor(grad_cafe_path)
    analysis.generate_distribution_by_result_sciences(True)