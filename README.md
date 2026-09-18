# Supervised Machine Learning - Cardiovascular Disease Prediction
This project applied supervised machine learning to clinical patient data to predict the presence or absence of heart disease. The project combined data wrangling, feature scaling, and binary classification using logistic regression to identify at-risk patients based on the demographic and cardiovascular profiles.

Early detection of heart disease presents an important medical challenge wherein multiple physiological factors interact. Traditional diagnostics can flag individual risk factors, but may fail to calulate the cumulative probability of disease onset based on a patients complete and complex clinical profile.

## Setup

### Prerequisits
- Python version: 3.14
- Dependencies: pip install pandas numpy matplotlib random scikit-learn

### Data Access
The project utilises a curated dataset (heartdisease.csv) that was originally provided as part of an academic coursework assignment. 

While derived from a public-access dataset, this specific dataset is a modified subset with certain variables and observations removed for educational purposes. As the exact licensing of this modified derivative is unconfirmed, the data file is not hosted publically in this repository.

To run the code locally:
1. Obtain a compatible heart disease dataset (such as the UCI Heart Disease dataset) and save it as heartdisease.csv.
2. Ensure your dataset contains the required clinical features (age, sex, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, presence of angina, and peak heart reading) and target variable (heart disease).
3. You may need to adjust the data wrangling steps in the code depending on the exact formatting of your alternative dataset.

### Execution
Clone this repository and run the main Python script from your terminal

## Workflow
1. Data wrangling: Filter missing or invalid zero-values in vital metrics and apply encoding from string to categorical variables to ensure inputs are readable. Split the data into a 70/30 train-test configuration and standardise continuous physiological features using z-score scaling to prevent larger numerical values from dominating the model.
2. Model training: Use a logistic regression classifier optimised over 1000 iterations to map the linear relationships between the clinical predictors and binary heart disease outcome.
3. Model prediction: Assess diagnostic performance on unseen test data
4. Model evaluation: Evaluate model performance on the unseen test data using a confusion matrix to visualise the rates of correct and incorrect disease classifications.
   
## Interpretation
### Data Wrangling
Structural anomalies were removed (such as resting blood pressure or cholesterol of 0) to ensure they did not skew the model. After cleaning the dataset comprised of 746 instances (decreased from 918 prior to model cleaning). Exploration of the data showed that patients had an average age of 53.

### Model Training and Prediction
Fitting a logistic regression model to the scaled training data mathematically isolates how different predictors influence cardiovascular risk. The model calculates a weighted probability for each patient using their specific combination of physiological readings rather than relying on a single isolated metric.

### Model Evaluation
A confusion matrix determines the rate or true and false positives and negatives in order to understand the model's sensitivity and specificity. It shows strong true negative (98) and good true positive (76) rates, and medium false negative (26) and false positive (24) rates.

## Next Steps
1. To introduce data reduction steps using feature selection to choose the most important features contributing to model performance and reduce noise.
2. To introduce hyperparameter tuning to enhance model accuracy and generalisation on unseen data.
3. To evaluate non-linear ensemble algorithms to capture complex interactions between clinical variables and enhance model transparency for clinical interpretation.
