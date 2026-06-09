# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
import random

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Import Data
Data_CVD = pd.read_csv("heartdisease.csv")

# Basic Data Exploration
print(Data_CVD)
Data_CVD.info()

# Organisation
Data_CVD[["RestingBP","Cholesterol"]] = (
    Data_CVD[["RestingBP","Cholesterol"]].replace(0, np.nan)
)
Data_CVD = Data_CVD.dropna()
Data_CVD['Sex'] = Data_CVD['Sex'].astype('category')
print(Data_CVD['Sex'].cat.categories)
Data_CVD['RestingECG'] = Data_CVD['RestingECG'].astype('category')
print(Data_CVD['RestingECG'].cat.categories)
Data_CVD['Angina'] = Data_CVD['Angina'].astype('category')
print(Data_CVD['Angina'].cat.categories)

# Advanced Data Exploration
Age_Desc = print(Data_CVD.describe())
Age_Sex_Desc = (Data_CVD.groupby('Sex')['Age']
           .agg(mean = 'mean', sd = 'std')
           .reset_index()
           )
print(Age_Sex_Desc)
Data_CVD['HeartDisease'].value_counts(normalize = True)


# PREPROCESSING

# Randomisation
random.seed(123)

# Split Sets
Observation = len(Data_CVD)  # Find number observations
Test_IDX = np.random.choice(
    Observation,
    size = round(Observation * 0.3),
    replace = False
)  # Randomly select 30% row indices
Test_Data = Data_CVD.iloc[Test_IDX]  # Attribute random 30% to test
Train_Data = Data_CVD.drop(Data_CVD.index[Test_IDX])  # Drop random 30% and attribute rest to train
print(Test_Data.info)  # Check no. test observations correct
print(Train_Data.info)  # Check no. train observations correct