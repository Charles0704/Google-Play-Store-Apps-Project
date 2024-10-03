# Machine Learning Approaches Towards Google Play Store Apps:

## $[Project$ $Background]:$

This project analyzes the Android market based on 10,000+ Google Play App data and 60,000+ customer review data provided by Kaggle, and provides decision-making suggestions to Android App developers to improve the quality of the user experience.

## $[Data$  $Processing$  $and$  $Feature$  $Engineering]:$   

Use Python(Pandas) to clean missing values, convert data types and standardize numerical features.  
Use DBSCAN, Isolated forest to detect outliers and remove them.

## $[Data$ $Visualization]:$ 
Use visualization tools such as Matplotlib and Seaborn to create box plots, histograms and conduct exploratory analysis to determine the relationship between Google Apps Ratings and predictive features: User reviews, Size, Number of installs, Price, Category, and so on.

## $[Model$ $Analysis]:$  

XGBoost:   
Accuracy: 0.7161246612466124  
Precision: 0.6986163771058592  
Recall: 0.7161246612466124  
F1 Score: 0.6873647586010466  

LightGBM： 
Accuracy: 0.7073170731707317  
Precision: 0.6787360283182414  
Recall: 0.7073170731707317  
F1 Score: 0.6772317670797979  

Catboost:   
Accuracy: 0.717479674796748  
Precision: 0.6918559839606064  
Recall: 0.717479674796748  
F1 Score: 0.6849244999049675  

### $[Dataset]: $ 
Google Play Store Apps Dataset
Link：https://www.kaggle.com/datasets/lava18/google-play-store-apps
