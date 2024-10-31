# Capstone project: Machine Learning Approaches Towards Google Play Store Apps

## Project Background:

This project analyzes over 10,000 Google Play apps and 60,000 customer reviews from Kaggle data, offering strategic insights for Android developers to enhance user experience quality  
Link：https://www.kaggle.com/datasets/lava18/google-play-store-apps

## Data Processing and Feature Engineering: 

Use Python(Pandas) to clean missing values, convert data types, standardize numerical features and encode categorical features.  
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

## NLP part:   
![output](https://github.com/user-attachments/assets/1d64fd2b-185c-4157-82e6-3ca69ed9bf2e)

