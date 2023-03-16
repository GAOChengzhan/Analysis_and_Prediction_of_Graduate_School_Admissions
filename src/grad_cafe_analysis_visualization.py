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

        fig.suptitle('GradCafe Dataset Labels Distribution Boxplots')

        # GRE Scores
        sns.boxplot(y=self.grad_cafe_df['gre'], ax=axes[0], color='#738bd9')

        # GPA 
        sns.boxplot(y=self.grad_cafe_df['gpa'], ax=axes[1], color='#e5a089')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_boxplots.png')

        plt.show()

    def generate_distribution_plots(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        # Remove outliers
        grad_cafe_prettify = self.grad_cafe_df[(np.abs(scipy.stats.zscore(self.grad_cafe_df['gpa'])) < 3)]
        grad_cafe_prettify = self.grad_cafe_df[(np.abs(scipy.stats.zscore(self.grad_cafe_df['gre'])) < 3)]

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        fig.suptitle('GradCafe Dataset Labels Distribution Histograms')

        # GRE Scores
        sns.histplot(data=grad_cafe_prettify, x='gre', ax=axes[0], bins=15, linewidth=0.3, edgecolor='white', color='#738bd9')

        # GPA
        sns.histplot(data=grad_cafe_prettify, x='gpa', ax=axes[1], bins=15, linewidth=0.3, edgecolor='white', color='#e5a089')

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

        corr = self.grad_cafe_df[['gpa', 'gre']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap='coolwarm', ax=axes, annot=True, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'color':'w'})
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

        plt.figure(figsize=(15, 8))
        axes = plt.axes()

        sns.regplot(x='gpa', y='gre', data=self.grad_cafe_df, fit_reg=True, ax=axes, color='#a7a2e8')
        axes.set_title('GradCafe Dataset Labels Correlation Matrix')

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
        result_counts.plot.pie(wedgeprops = {'linewidth': 1, 'edgecolor': 'white' })

        plt.title('GradCafe Dataset Application Result Pie Chart')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_application_result_chart.png')

        plt.show()

    def generate_averages_by_application_results(self, export=False):
        '''
        Generate the plot on the average gpa and gre for each application result
        '''

        assert isinstance(export, bool)

        result_by = self.grad_cafe_df.groupby('notif_result')[['gpa', 'gre']].agg('mean')
        result_by = result_by.reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.subplots_adjust(hspace=0.25, wspace=0.5)

        fig.suptitle('GradCafe Dataset Averages By Application Result')

        # Distribution of gre by result
        sns.barplot(x='notif_result', y='gre', data=result_by, ax=axes[0], color='#738bd9').set_yscale('log')

        # Distribution of gpa by result
        sns.barplot(x='notif_result', y='gpa', data=result_by, ax=axes[1], color='#e5a089').set_yscale('log')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_averages_by_application_result.png')

        plt.show()

    def generate_top_ten_majors(self, export=False):
        '''
        Generate a frequency plot of top ten universities
        '''

        assert isinstance(export, bool)

        plt.figure(figsize=(20, 20))

        # top 10
        freq = self.grad_cafe_df['major'].value_counts().nlargest(10)

        plt.title('Top 10 most popular major')
        plt.bar(freq.index, freq.values, color=(0.2, 0.4, 0.6, 0.6))
        plt.xticks(rotation=90)
        plt.xlabel('Major')
        plt.ylabel('Frequency')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_top_ten_majors.png')

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

        # Remove outliers
        engineering_df_prettify = self.engineering_df[(np.abs(scipy.stats.zscore(self.engineering_df['gpa'])) < 3)]
        engineering_df_prettify = self.engineering_df[(np.abs(scipy.stats.zscore(self.engineering_df['gre'])) < 3)]

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        fig.suptitle('GradCafe Dataset Labels Distribution By Engineering Major')

        # GRE Scores
        sns.histplot(data=engineering_df_prettify, x='gre', ax=axes[0], bins=15, linewidth=0.3, edgecolor='white', color='#738bd9')

        # GPA
        sns.histplot(data=engineering_df_prettify, x='gpa', ax=axes[1], bins=15, linewidth=0.3, edgecolor='white', color='#e5a089')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_engineering_distribution_plots.png')

        plt.show()

    def generate_distribution_plots_sciences(self, export=False):
        '''
        Generate distribution summary of the GradCafe dataset by sciences major

        :param export: select True to export the boxplot pictures, else False
        '''

        assert isinstance(export, bool)

        # Remove outliers
        sciences_df_prettify = self.sciences_df[(np.abs(scipy.stats.zscore(self.sciences_df['gpa'])) < 3)]
        sciences_df_prettify = self.sciences_df[(np.abs(scipy.stats.zscore(self.sciences_df['gre'])) < 3)]

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        fig.suptitle('GradCafe Dataset Labels Distribution By Sciences Major')

        # GRE Scores
        sns.histplot(data=sciences_df_prettify, x='gre', ax=axes[0], bins=15, linewidth=0.3, edgecolor='white', color='#738bd9')

        # GPA
        sns.histplot(data=sciences_df_prettify, x='gpa', ax=axes[1], bins=15, linewidth=0.3, edgecolor='white', color='#e5a089')

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
        fig.subplots_adjust(hspace=0.25, wspace=0.5)

        fig.suptitle('GradCafe Dataset Averages By Engineering Major Application Result')

        # Distribution of gre by result
        sns.barplot(x='notif_result', y='gre', data=engineering_result_by, ax=axes[0], color='#738bd9').set_yscale("log")

        # Distribution of gpa by result
        sns.barplot(x='notif_result', y='gpa', data=engineering_result_by, ax=axes[1], color='#e5a089').set_yscale("log")

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
        fig.subplots_adjust(hspace=0.25, wspace=0.5)

        fig.suptitle('GradCafe Dataset Averages By Sciences Major Application Result')

        # Distribution of gre by result
        sns.barplot(x='notif_result', y='gre', data=sciences_result_by, ax=axes[0], color='#738bd9').set_yscale("log")

        # Distribution of gpa by result
        sns.barplot(x='notif_result', y='gpa', data=sciences_result_by, ax=axes[1], color='#e5a089').set_yscale("log")

        if export:
            plt.savefig('analysis_outputs/grad_cafe_sciences_distribution_by_result.png')

        plt.show()

class ByTopTenUniversity(GradCafeAnalysis):

    def __init__(self, grad_cafe_path):
        super().__init__(grad_cafe_path)

        # Count the top frequency of each institution in the 'institution' column
        top10, data = self.grad_cafe_df['institution'].value_counts().head(10), {}

        # Loop through the top 10 institutions and compute the average GRE and GPA for each one
        for institution in top10.index:
            institution_df = self.grad_cafe_df.loc[self.grad_cafe_df['institution'] == institution]
            avg_gpa = institution_df['gpa'].mean()
            avg_gre = institution_df['gre'].mean()
    
            data[institution] = {'avg_gpa': avg_gpa, 'avg_gre': avg_gre}

        # Convert the data dictionary to a DataFrame
        self.top10_df = pd.DataFrame(data).T

    def generate_top_ten_universities(self, export=False):
        '''
        Generate a frequency plot of top ten universities
        '''

        assert isinstance(export, bool)

        plt.figure(figsize=(20, 20))

        # Top 10
        freq = self.grad_cafe_df['institution'].value_counts().nlargest(10)

        plt.title('Top 10 Most Popular Schools')
        plt.bar(freq.index, freq.values, color=(0.2, 0.4, 0.6, 0.6))
        plt.xticks(rotation=90)
        plt.xlabel('Institution')
        plt.ylabel('Frequency')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_top_ten_schools.png')

        plt.show()

    def generate_top_ten_universities_gre(self, export=False):
        '''
        Generate a top ten universities average GRE
        '''

        assert isinstance(export, bool)

        self.top10_df.plot(kind='line',y=['avg_gre'], color='#8be0cb')
        plt.xticks(range(len(self.top10_df.index)), self.top10_df.index, rotation=90)
        plt.xlabel('Institution')
        plt.ylabel('Average Value')
        plt.title('Average GRE for Top 10 Institutions')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_top_ten_schools_avg_gre.png')

        plt.show()

    def generate_top_ten_universities_gpa(self, export=False):
        '''
        Generate a top ten universities average GPA
        '''

        assert isinstance(export, bool)

        self.top10_df.plot(kind='line',y=['avg_gpa'], color='#8be0cb')
        plt.xticks(range(len(self.top10_df.index)), self.top10_df.index, rotation=90)
        plt.xlabel('Institution')
        plt.ylabel('Average Value')
        plt.title('Average GPA for Top 10 Institutions')

        if export:
            plt.savefig('analysis_outputs/grad_cafe_top_ten_schools_avg_gpa.png')

        plt.show()

# Main start here, for testing
if __name__ == '__main__':
    grad_cafe_path = 'data/cleaned_data/cleaned_grad_cafe_admissions.csv'

    analysis = ByTopTenUniversity(grad_cafe_path)
    analysis.generate_top_ten_universities_gpa(True)