# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pandas._config import display

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Import Data
ds = pd.read_csv("heartdisease.csv")

# Basic Data Exploration
print(ds)
ds.info()

# Organisation
ds[["RestingBP","Cholesterol"]] = (
    ds[["RestingBP","Cholesterol"]].replace(0, np.nan)
)
ds = ds.dropna()

# Convert type
ds['Sex'] = ds['Sex'].astype('category')  # Convert to categorical
ds['RestingECG'] = ds['RestingECG'].astype('category')
ds['Angina'] = ds['Angina'].astype('category')

ds = pd.get_dummies(ds,
                    columns = ["Sex", "RestingECG", "Angina"],
                    drop_first = True
                    )

bool_col = ['Sex_M', 'RestingECG_Normal', 'RestingECG_ST', 'Angina_Y']
for col in bool_col:
    ds[col] = ds[col].astype(int)

print(ds.dtypes)

# Advanced Data Exploration
print(ds.describe())
age_sex_desc = (
    ds.groupby("Sex_M")["Age"].agg(mean = "mean", sd = "std").reset_index()
)
print(age_sex_desc)

ds['HeartDisease'].value_counts(normalize = True)


# PREPROCESSING

# Test-Train Split
random.seed(123)

obs = len(ds)
test_idx = np.random.choice(
    obs,
    size = round(obs * 0.3),
    replace = False
)

test_ds = ds.iloc[test_idx]
train_ds = ds.drop(ds.index[test_idx])

print(test_ds.info)
print(train_ds.info)

# Scaling
scale = StandardScaler()

scaled_train = train_ds.copy()
scaled_train[["Age", "RestingBP", "MaxHR", "HeartPeakReading"]] = scale.fit_transform(train_ds[["Age", "RestingBP", "MaxHR", "HeartPeakReading"]])
print(scaled_train)


scaled_test = test_ds.copy()
scaled_test[["Age", "RestingBP", "MaxHR", "HeartPeakReading"]] = scale.transform(test_ds[["Age", "RestingBP", "MaxHR", "HeartPeakReading"]])
print(scaled_test)

# TRAINING

# Set variables
train_predict = scaled_train[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "HeartPeakReading", "Sex_M", "RestingECG_Normal", "RestingECG_ST", "Angina_Y"]]
train_outcome = scaled_train["HeartDisease"]

# Train model
model = LogisticRegression(max_iter = 1000)
model.fit(train_predict, train_outcome)


# PREDICT

test_predict = scaled_test[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "HeartPeakReading", "Sex_M", "RestingECG_Normal", "RestingECG_ST", "Angina_Y"]]
test_outcome = scaled_test["HeartDisease"]

prediction = model.predict(test_predict)

print(prediction)