#  Machine Learning Approaches Towards Google Play Store Apps

## Background:
This repository contains the capstone project of ESE527, which analyzes 10,000+ Google Play apps info information data and 60,000+ customer reviews from Kaggle, offering strategic insights for Android developers to enhance user experience quality. In this project, We apply **XGBoost**, **LightGBM**, and **CatBoost** to Google-play-store-apps dataset to classify and predict the rating level of different Apps. We also apply **Bert**(Bidirectional encoder representations from transformers) model to conduct sentiment analysis on customer reviews.  
Dataset Link：https://www.kaggle.com/datasets/lava18/google-play-store-apps

## Environment:
```python version：3.12.4```  
```PyTorch version:  2.4.1+cu124```  
```CUDA Version：12.5```

## Data Processing and Feature Engineering: 

Utilized Python (Pandas) for data cleaning, including handling missing values, data type conversion, numerical feature standardization, and categorical encoding.  
Applied DBSCAN and Isolation Forest for outlier detection and removal.

## Data Visualization: 
Leveraged Matplotlib and Seaborn for data visualization, creating box plots, histograms, Heatmap, and conducting exploratory analysis to examine relationships between Google App ratings and predictive features such as user reviews, app size, install count, price, category, and more.


<img src="C:\Users\Chuxu\Desktop\output" height="1150px" width="1240px"/>



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

