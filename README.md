# Capstone project: Machine Learning Approaches Towards Google Play Store Apps

## Project Background:

This project analyzes the Android market based on 10,000+ Google Play App data and 60,000+ customer review data provided by Kaggle, and provides decision-making suggestions to Android App developers to improve the quality of the user experience.

## Data Processing and Feature Engineering: 

Use Python(Pandas) to clean missing values, convert data types and standardize numerical features.  
Use DBSCAN, Isolated Forest for outlier detection and removal.

## Data Visualization: 
Use visualization tools such as Matplotlib and Seaborn to create box plots, histograms and conduct exploratory analysis to determine the relationship between Google Apps Ratings and predictive features: User reviews, Size, Number of installs, Price, Category, and so on.

## Model Analysis Result:  

### XGBoost:   
Accuracy: 0.716  
Precision: 0.699  
Recall: 0.716  
F1 Score: 0.687   

### LightGBM：   
Accuracy: 0.707  
Precision: 0.679  
Recall: 0.707  
F1 Score: 0.677  

### Catboost:   
Accuracy: 0.717  
Precision: 0.692  
Recall: 0.717  
F1 Score: 0.685  

## $[NLP$ $part]:$   

### $[Dataset]: $ 
Google Play Store Apps Dataset
Link：https://www.kaggle.com/datasets/lava18/google-play-store-apps
