# TwitterAnalytics

The Twitter Analytics Tool can used to do political stance prediction for Twitter users.
It's findings can be used to analyse political participation in hashtags.


Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

 

The project comes with all things you need to do your own AI-based stance analysis. Such as:

1.  Postgres SQL DWH creation
2.  Required ETL operations
3.  Download of entire Hashtags
4.  Download of Twitter followers and friends
5.  BERT Machine learning training
6.  BERT Inference
7.  Confidence calculation

The technique can be used to analyse 

Not included:
Evaluation Set
DB

 







Udacitiy Spark Capstone Project
Feature detection and machine learning in a big data environement
About the Project
Purpose of the project is to identify users in a dataset, that are likely to unsubscribe from a payed service. The dataset contains user data from an anonymised music streaming platform. The projects focus is on feature detection in a large dataset.
Full dataset is 12GB but in this project a 128 MB subset is used. Please also note the blog post for this project on medium.com.

Libraries
pyspark
pandas
matplotlib
sklearn
Files
main.py - Regular python code Sparkify.ipynb - Jupyter Notebook with structure as suggested during Udacity project

Usage
Download files and run in Python 3.8.
Make sure to provide "mini_sparkify_event_data.json" in same project folder as main.py The file can not be provided in Github due to file size Limiation of 25 MB!
Run Juypter notebook OR run main.py (no parameters required)
Note: I deliberately used only three machine learnig algoriths in this project. This is a learning from last project that had a pipeline so big it took 8 hours to run

Processing Steps
Step1: Exploratory Data Analysis

Page 'Cancellation Confirmation' was selected from Spark Dataset as churn.
Unused data exploration steps were removed during code cleaning Step2: Feature Creation
In this step the data is first enriched with a week column.
Also a feature is defined as average songs listened to per week and session.
Both is combined using a pivot table creating a matrix with [average songs listened to per week and session] as value and weeks as columns.
This pivot table is combined with user data, indicating which user cancelled subscription
Eventually the matrix in split into a features and labels to prepare for machine learning training. Step3: Selected Models
In the modelling part a print_predictions() function is defined.
It is later used to return results of each model. Three models were chosen randomly. I stopped at three to keep runtime low (in project 3 I used a pipeline with all sort of algorithms including grid search and got a training time of 8 hours).
Results
Of each of the three used algorithms Random Forrest delivers the best performance. Precision and recall score however are rather low. A bigger training dataset and more features could possibly lead to better results.

** Random Forrest**

[[33 2] [ 3 8]] precision recall f1-score support

     0.0       0.92      0.94      0.93        35
     1.0       0.80      0.73      0.76        11

accuracy                           0.89        46
macro avg 0.86 0.84 0.85 46 weighted avg 0.89 0.89 0.89 46
