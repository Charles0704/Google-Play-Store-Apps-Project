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

1.We categorize Rating into 2 class：  
If Rating $\in$[1, 4)   ---> 0  
If Rating $\in$[4, 5)   ---> 1  

2.Use One-hot Encoding to convert categorical features(Category, Content Rating, Genres, Type) into numerical features  

#### Outlier Removal:   
Applied DBSCAN and Isolation Forest for outlier detection and removal.


## Data Visualization: 
Leveraged Matplotlib and Seaborn for data visualization, creating box plots, histograms, Heatmap, and conducting exploratory analysis to examine relationships between Google App ratings and predictive features such as user reviews, app size, install count, price, category, and more.

<img src=https://github.com/user-attachments/assets/e3c7c857-cf4b-4b5b-952f-5769f177ccc7 height="400px" width="800px"/>

## Model Analysis Result:
#### XGBoost:   
<img src = https://github.com/user-attachments/assets/142520af-f8ef-4960-82c7-5ae0fa1ad377  height="400px" width="400px"/>

#### LightGBM：   
<img src = https://github.com/user-attachments/assets/9d8393c1-285f-4ee3-8655-3142261db842  height="400px" width="400px"/>

#### Catboost:   

<img src = https://github.com/user-attachments/assets/fbeaa8a9-cdf1-4053-ac8f-8c0be37cd11e  height="400px" width="400px"/>

## Sentiment Analysis Part:   

We choose 10000 lines of customer reviews as a sample and extract reviews with sentence length largeer than 30, then we divide the samples into training and testing sets.  

Here is the result of Bert with 20 epochs:
![6bbd9d4c8b85a99dc3840f65e6d3976](https://github.com/user-attachments/assets/a14fd662-0664-47d4-86b0-fd9e18959af0)


![Figure_1](https://github.com/user-attachments/assets/7a6397d9-7912-4718-a4c7-e9aaaf253d70)



