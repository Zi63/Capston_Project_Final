# Capston_Project_Final
Startup Investment &amp; Success Prediction 
# Startup Investment and Success Prediction



# Introduction:

This repository is dedicated to the development and application of machine learning algorithms aimed at forecasting the success of startup ventures. By analyzing critical factors including funding amounts, geographical location, industry sectors, and team sizes, this project seeks to empower both investors and entrepreneurs with predictive insights to guide their decisions on funding, launching, or evaluating the potential success of startups.

# Data Source

The core dataset originates from crunchbase.com, encompassing details on numerous startups such as their funding stages, geographic locations, industry categories, team compositions, among other pertinent attributes. Prior to analysis, this dataset undergoes a rigorous cleaning and preprocessing routine to address missing values, eliminate outliers, and transform categorical variables into a machine-readable numeric format.
Machine Learning Approach



# Objective of Analysis

The primary objective of our analysis is to uncover patterns and factors that contribute to the success or failure of startups. By exploring relationships between different variables, such as funding rounds, participant averages, and the geographical distribution of these startups, we aim to generate actionable insights that could guide entrepreneurs and investors alike. The project will involve several key phases, including data preprocessing, exploratory data analysis (EDA), feature engineering, and the development of predictive models to forecast startup outcomes.

Our predictive model experiments with a variety of machine learning algorithms, including but not limited to the XGBoost Classifier (XBC), AdaBoost Classifier (ABC), Random Forest Classifier (RFC), and Gradient Boosting Classifier (GBC). A comprehensive grid search methodology is employed across these models to identify and optimize hyper parameters, thereby enhancing model accuracy.
Findings
The exploration revealed that the Random Forest Classifier (RFC) outperformed other models, delivering an accuracy score of 0.85. Furthermore, an analysis on feature importance was conducted, highlighting funding and industry as the key determinants in predicting startup success.

---------------------------------------------------------------------------------------------------------------------
## Project Organization

## Data Overview
   
   Dataset Name: (startup data.csv)From Kaggle + Will use multiple dataset for this Project
   Description: This dataset offers a comprehensive look into various startups, providing details on their funding history, foundational information, and 		   
                operational status. This data is instrumental in analyzing patterns and trends in the startup ecosystem, offering valuable 
                insights into the factors that contribute to the success or failure of new ventures.
   Key Features: 
              first_funding_at: The date of the first funding received.
              founded_year: The year the company was founded.
              object_id: A unique identifier for the company.
              age_first_funding_year: The age of the company when it received its first funding.
              closed_at: The date when the company closed (if applicable).
              status_encoded: An encoded form of the company's current status.
              funding_total_usd: The total funding received in USD.
              state_code: The state code where the company is located.
              Unnamed: 6: This might be an unnamed column from the original dataset.
              last_funding_at: The date of the last funding received.
              Unnamed: 0: Another unnamed column from the dataset.
              age_last_milestone_year: The age of the company at the last milestone.
              duration_to_first_funding: Duration from founding to first funding.
	      status: The current status of the company (e.g., operating, closed).
	      age_first_milestone_year: The age of the company at the first milestone.
	      founded_month: The month the company was founded.
 	      founded_at: The exact date the company was founded.
  	      age_last_funding_year: The age of the company when it received its last funding.
   
   Analysis Goals
             To understand the factors that contribute to the success or failure of startups
             To identify trends and patterns in startup funding 
             To explore the impact of geographical location on startup success
             Predictive modeling: Developing a model that can predict the potential success or failure of a startup based on various input features.

   Tools and Technologies Used
      	     Programming Languages and Libraries: Python
             Libraries:
             pandas: Data manipulation and analysis
             seaborn, matplotlib.pyplot: Data visualization
             sklearn (various modules): Machine learning and data preprocessing
   
   Repository Structure
    
            /data: Directory for datasets
	    /notebooks: Jupyter notebooks  
            /scripts:  
            /reports: Analysis reports or insights

# Flowchart Data Analysis Workflow

      	    Data Collection and Loading
            Data Cleaning and Preprocessing
            Exploratory Data Analysis
            Feature Engineering
	    Model Building
	    Model Evaluation
	    Interpretation of Results
-------------------------------------------------------------------------------------------------------------------


# How to Execute the Code
Below are the steps to run the project code:
Terminal/bash

 
# Steps for execution:
 
#1. Install required libraries: 
- pandas 
- numpy 
- seaborn 
- scikit-learn 
- plotly 
 
#2. Load the dataset from "startup data.csv" into a pandas dataframe named 'dataset'. 

#3. Modify the 'status' column: rename to 'is_acquired' and recode 'acquired' as '1', 'operating' as '0'. 

#4. Generate a heatmap to visualize correlations among dataset features. 

#5. Address outliers by computing the interquartile range (IQR) and excluding values outside [Q1 - 1.5IQR, Q3 + 1.5IQR]. 
# 6. Impute missing values in numerical features using the KNNImputer class from scikit-learn. 
# 7. Convert categorical features to numerical format and remove non-essential features. 
# 8. Construct a correlation matrix and exclude features with a correlation coefficient below 0.2 with 'is_acquired'. 
# 9. Implement auxiliary functions like 'ignore_warn', 'draw_heatmap', 'getOutliersMatrix', and 'imputing_numeric_missing_values' to facilitate the preprocessing steps. 
# 10. Prepare the cleaned and preprocessed data for analysis and store it back into the 'dataset' dataframe. 
# 11. Apply a meta-modeling strategy that leverages the strengths of multiple base models (XBC, ABC, RFC, GBC) to refine accuracy. 
# 12. Conduct grid search for each model to tune hyperparameters for optimal performance. 
# 13. Display grid search outcomes, selecting the best classifiers for ensemble modeling. 
# 14. Employ Plotly to craft a scatterplot visualizing the feature importance as determined by each model.



# Limitations and Directions for Future Work:
This project acknowledges certain limitations, including potential biases within the dataset and the constrained range of features considered for success prediction. Future endeavors may extend this research by incorporating more diverse datasets, exploring additional predictive features, and validating the model's efficacy with actual startup outcomes.


 



