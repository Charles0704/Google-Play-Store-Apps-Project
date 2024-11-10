#  Machine Learning Approaches Towards Google Play Store Apps

## Background:
This repository contains the capstone project of ESE527, which analyzes 10,000+ Google Play apps info information data and 60,000+ customer reviews from Kaggle, offering strategic insights for Android developers to enhance user experience quality. In this project, We apply **XGBoost**, **LightGBM**, and **CatBoost** to Google-play-store-apps dataset to classify and predict the rating level of different Apps. We also apply **Bert**(Bidirectional encoder representations from transformers) model to conduct sentiment analysis on customer reviews.  

## Dataset:
Dataset Link：https://www.kaggle.com/datasets/lava18/google-play-store-apps

## Environment:
```python version：3.12.4```  
```PyTorch version:  2.4.1+cu124```  
```CUDA Version：12.5```

## Data Processing and Feature Engineering: 

#### Data Processing：  
Utilized Python (Pandas) for data cleaning, including handling missing values, data type conversion, numerical feature standardization, and categorical encoding.  

#### Feature Engineering：  

1.We categorize Rating into 3 class：  
If Rating $\in$[1, 4)   ---> 0  
If Rating $\in$[4, 4.5) ---> 1  
If Rating $\in$[4. 5] ---> 2  

2.Use One-hot Encoding to convert categorical features(Category, Content Rating, Genres, Type) into numerical features  

#### Outlier Removal:   
Applied DBSCAN and Isolation Forest for outlier detection and removal.


## Data Visualization: 
Leveraged Matplotlib and Seaborn for data visualization, creating box plots, histograms, Heatmap, and conducting exploratory analysis to examine relationships between Google App ratings and predictive features such as user reviews, app size, install count, price, category, and more.


<img src=https://github.com/user-attachments/assets/e3c7c857-cf4b-4b5b-952f-5769f177ccc7 height="400px" width="800px"/>


## Model Analysis Result:(Still Working...)  
#### XGBoost:   


#### LightGBM：   


#### Catboost:   


## Sentiment Analysis Part:   
Here, I choose 10000 lines of customer reviews as a sample and extract statements with sentence length largeer than 30, then I divide the samples into training and testing sets
![Figure_1](https://github.com/user-attachments/assets/0555dc60-2d46-4597-ad9f-dc90186b1ec0)


