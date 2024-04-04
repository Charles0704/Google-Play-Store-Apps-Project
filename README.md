ESE-527-Project:

Title： Machine Learning Approaches Towards Google Play Store Apps

Chuxuan He, Haipeng Zhao

【Project Background】: This project analyzes the Android market based on 10,000+ Google Play App data and 60,000+ customer review data provided by Kaggle, and provides decision-making suggestions to App developers to improve the quality of the user experience.
【Data Processing and Feature Engineering】: Use Python to clean missing values, convert data types and standardize numerical features. Use DBSCAN to detect outliers and remove them.
【Data Visualization】: Use visualization tools such as Matplotlib and Seaborn to create box plots ,histograms and conduct exploratory analysis to determine the relationship between Google Apps ratings and predictive factors: User reviews, Number of downloads, Price and so on.
【Model Analysis】: Use LightGBM, Catboost, SVM and other ML models to classify and predict Google Apps ratings and find the optimal parameters through “Optuna”. The accuracy of LightGBM is about 52%

Dataset： Google Play Store Apps Dataset
Link：https://www.kaggle.com/datasets/lava18/google-play-store-apps
