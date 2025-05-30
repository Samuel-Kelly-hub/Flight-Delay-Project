﻿# Flight-Delay-Project
***
This is a python-based project to predict the flight delays of US domestic flights.  It is using historical data from 2015 from (https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv).

## Overview and Structure:

This project is contained within 3 Jupyter Notebooks, each one exemplifying a different subset of skills.  The notebooks are:
- Data_Cleaning.ipynb
- Exploratory_Data_Analysis.ipynb
- Flight_Predictions.ipynb

The 1st notebook, Data_Cleaning, is focused on 2 main problems.  First, the identifiers for the airports are heterogeneous, having both the standard IATA codes as well as anonymous 5-digit codes.  I resolved the problem by finding what the 5-digit codes are (internal codes for the US Government), and then used string methods, regular expressions, and data engineering to match and replace the 5-digit codes.  Second, the time columns were in varying time zones, and formats.  I performed timestamp reconstruction and applied feature engineering to standardise the columns into a machine-readable datetime format whilst maintaining much useful information as possible for downstream machine learning.

The 2nd notebook, Exploratory_Data_Analysis, examines the data that was cleaned in Data_Cleaning.  Through the incorporation of data visualisation, using Matplotlib and Seaborn (on top of statistical methods) I was able to identify relationships and variations within the data.  I incorporated a variety of Dataframes and groupings to uncover key patterns in the data.  This notebook uncovered and elucidated key variations in the data that would inform my Feature Engineering including whether particular numerical features would be treated as categorical or continuous.

In order to handle large amounts of categorical data, the 3rd notebook, Flight_Predictions, uses several Linear Regression and Gradient Boost models to predict the arrival delay of flights: LinearRegression and ElasticNet from Scikit-learn, along with XGBRegressor, LGBMRegressor, and CatBoostRegressor.  I created several derivate features to try and explain variations in the data found in Exploratory_Data_analysis.  The notebook includes descriptions of how the models work as well as feature analysis and the use of a meta-model to incorporate the different patterns found both by Linear Regression and Gradient Boost.

## Skills Demonstrated in Each Notebook:

**Data_Cleaning.ipynb**
- Data Engineering
- String Manipulation
- Dealing with Missing Values
- Data Standardisation
- Dataset Reconcilisation

**Exploratory_Data_Analysis.ipynb**
- Data Visualisation
- Statistical Analysis
- Reshaping Data
- Data Analysis
- Feature Interaction Analysis

**Flight_Predictions.ipynb**
- Feature Engineering
- Sparse Data
- Data Normalisation
- An Understanding of how various Machine Learning Algorithms Work
- Machine Learning Optimisation
- Feature Importance Analysis


## Libraries Used:
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Seaborn
- Scipy

## How to Run the Application:
- Clone this repository.
- Make sure you have the correct packages installed.
- Unzip any zipped files.
- Open each Jupyter Notebook and run the cells in order.
