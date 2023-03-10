<!------------------------------------------ TITLE BLOCK --------------------------------------------------------------->
<h1 align="center"> Group 9: Analysis and Prediction of Graduate School Admissions </h1>

<p align="center">
    Course Project for UCSD ECE 143: Programming for Data Analytics
    <br /> <br />
    <a href="gchengzhan@ucsd.edu"> Chengzhan Gao </a>
    ·
    <a href="how016@ucsd.edu"> Houtianfu Wang </a>
    ·
    <a href="ken010@ucsd.edu"> Kendrick Nguyen </a>
    ·
    <a href="w8dong@ucsd.edu"> Weirong Dong </a>
    ·
    <a href="vpawar@ucsd.edu"> Varun Pawar </a>
</p>


<!------------------------------------------ TABLE OF CONTENTS ---------------------------------------------------------->
<details open="open">
  <summary><h2 style="display: inline-block"> Table of Contents </h2></summary>
  <ol>
    <li>
      <a href="#about-the-project"> About The Project </a>
      <ul>
        <li><a href="#datasets"> Datasets </a></li>
        <li><a href="#job-to-be-done"> Job To Be Done </a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started"> Getting Started </a>
      <ul>
        <li><a href="#prerequisites"> Prerequisites </a></li>
        <li><a href="#how-to-run"> How to Run </a></li>
      </ul>
    </li>
    <li><a href="#file-architecture"> File Architecture </a></li>
  </ol>
</details>


<!------------------------------------------ About The Project ---------------------------------------------------------->
## About The Project

Every student applying for a college education surely wants to know what are the chances of getting into their dream school. 
Our project is designed to visualize and predict the probability of admission of certain students and the universities to 
which they apply, which is beneficial for the students in need.

The results of this project are expected to help worldwide students who are applying for master’s programs. With our prediction 
as guidelines, they could apply a better application strategy if they already know for which kind of universities they are more 
attractive, so that they have a much bigger chance of admission.

### Datasets

The selected datasts includes parameters of interest, such as GRE Scores and Undergraduate GPAs, that are considered significant 
during graduate application processes.

* Graduate Admission Results via [GradCafe Statistics (scraped)](https://github.com/AlpAribal/gradcafestats/blob/master/data/submissions.csv)
* Graduate Admission Results via [Kaggle](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions?resource=download)

### Job To Be Done
1. Firstly, a data analysis will be carried out to explore the basic statistics and properties of the dataset, and build up the 
relationship among different parameters. 
2. Then based on the data analysis results, several prediction models (like logistic regression, naive bayes, support vector machine, 
nearest neighbor classifier, etc.) will be trained. 
3. A UI will be implemented that would allow users to input parameters (GRE scores, TOEFL scores etc.) to predict probability of 
admission using our trained models. 


<!------------------------------------------ Getting Started ---------------------------------------------------------->
## Getting Started
Unfortunately, the Graduate Admission Results via [GradCafe Statistics (scraped)](https://github.com/AlpAribal/gradcafestats/blob/master/data/submissions.csv)
dataset was too large to push to the repository. Download the dataset and place it inside the `data/raw_data` folder.

Now to get a local copy up and running follow these steps.

### Prerequisites
Clone the repository:
```
git clone https://github.com/solityde826/ECE143_Group9
```

Install the following dependencies:
* implicit
* Jupyter
* Matplotlib
* numpy
* openpyxl
* pandas
* PyTorch
* scikit-learn
* seaborn
* xgboost 

or alternatively run,
```
pip install -r requirements.txt
```

### How to Run
1. Run the `data_cleaning_wrangling.ipynb` notebook to obtain a clean and wrangled output
of the raw datasets
2. Run the `data_analysis_visualization.ipynb` notebook to analyze and visualize the datasets
3. Run the `prediction.ipynb` notebook to generate prediction models from the datasets


<!------------------------------------------ File Architecture  ---------------------------------------------------------->
## File Architecture
```
[ECE143_Group9]
├─ 📁data
    ├─ 📁cleaned_data
        ├─ 📄cleaned_grad_cafe_admissions.csv
        ├─ 📄cleaned_kaggle_grad_admissions.csv
        ├─ 📄unlabled_grad_cafe_admissions.csv
    ├─ 📁raw_data
        ├─ 📄kaggle_grad_admissions.csv
        ├─ 📄submissions.csv
├─ 📁src
    ├─ 📄data_analysis_visualization.ipynb
    ├─ 📄data_cleaning_wrangling.ipynb
    ├─ 📄prediction.ipynb
├─ 📄.gitignore
├─ 📄README.md
```