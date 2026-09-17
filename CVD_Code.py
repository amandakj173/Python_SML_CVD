# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
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
scaled_train = scale.fit_transform(train_ds)
scaled_test = scale.transform(test_ds)
print(scaled_train)
print(scaled_test)


# TRAINING

X = Train_Data[
    ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "HeartPeakReading", "Sex_M", "RestingECG_Normal", "RestingECG_ST", "Angina_Y"]
]  # Set predictors from training dataset
Train_Data[["Sex_M", "RestingECG_Normal", "RestingECG_ST", "Angina_Y"]] = Train_Data[["Sex_M", "RestingECG_Normal", "RestingECG_ST", "Angina_Y"]].astype(int)
X = sm.add_constant(X)  # Add intercept to predictors
Y = Train_Data["HeartDisease"]  # Set dependent variable
Train_Model = sm.GLM(Y, X, family = sm.families.Binomial())  # Fit GMM
Results = Train_Model.fit()
print(Results.summary())