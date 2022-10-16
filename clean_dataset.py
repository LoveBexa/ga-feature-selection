# Imbalanced learn library

pip install -U imbalanced-learn

import pandas as pd
import random, math
from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("poster")
plt.rcParams["figure.figsize"] = [8,6]

telecom_dataset = pd.read_csv("uni_data (3).csv")

telecom_dataset.head()

########### Convert CSV to DataFrame ########### 

# Show full width and height of dataframe
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)


# Count the churn and no churn records
data = telecom_dataset
data["Churn"].value_counts()

# Get info on total 'features' 

data.info()

import numpy as np

# Doesn't seem like there are any missing but I know there are as ? so lets look for that and replace with NaN
data = data.replace('?', np.nan)
data.head(10)



# Drops all missing values

# data = data.dropna() 

# Calculate all missing values again after filling ? with NaN

data.isna().sum()

# Look at Handset Price, Home Ownership and Marital Status in More Depth
high_missing = pd.DataFrame(data, columns=['Homeownership', 'HandsetPrice', 'MaritalStatus' ])

# Count unique values in each one
high_missing['Homeownership'].value_counts()

high_missing['HandsetPrice'].value_counts()
# Numerical data I was going to fill in the missing values with the average but it might skew the data model

# Over 55% Missing HandSet Price!!! Lets just drop the entire column

# data = data.drop(['HandsetPrice'], axis=1) 


# data = data.drop(['UnansweredCalls'], axis=1) 



data

high_missing['MaritalStatus'].value_counts()

# Convert all YES to 1 and all NO to 0
data_clean = data.replace({'Yes': 1, 'No': 0})	
data_clean

# Convert unknown/known to 0 and 1 
data_clean = data_clean.replace({'Known': 1, 'Unknown': 0})	
data_clean

# Convert Marital Status to int and NaN = 0 
data_clean['MaritalStatus'] = data_clean['MaritalStatus']
data_clean.head(10)

# Convert float in martial status to int
data_clean['MaritalStatus'] = data_clean['MaritalStatus'].fillna(0)
data_clean['MaritalStatus'].value_counts()

data_clean['CreditRating'].value_counts()

data_clean['PrizmCode'] = data_clean['PrizmCode'].replace({"Other": 0,  "Suburban": 1, "Town": 2,"Rural":3 })
data_clean['PrizmCode'].value_counts()

# Rename Occupation to Working yes/no 1/0
data_drop = data_clean.rename(columns={"Occupation": "Working"})

data_drop['Working'] = data_drop["Working"].replace({'Other': 0, 'Professional': 1,'Professiol': 1, 'Crafts': 1, 'Clerical': 1, 'Self': 1,'Student': 0,'Homemaker': 0,'Retired': 0 })	
data_drop['Working'].value_counts()

# Drop items with too many unique strings 

data_drop = data_drop.drop(['ServiceArea'], axis=1)

# Convert home ownership known = 1, NaN = 0
data_drop['Homeownership'] = data_drop['Homeownership'].replace({'Known': 1, 'Unknown': 0})	
data_drop['Homeownership'] = data_drop['Homeownership'].fillna(0)

# data_clean['Homeownership'] = data_clean['Homeownership'].astype(np.int64)
data_drop['Homeownership'].value_counts()

#  Remove everything EXCEPT numerical values in credit rating 
data_drop['CreditRating'] = data_drop['CreditRating'].str.extract('(\d+)', expand=False)
data_drop.head(10)

# # Replace all 0 in handset price to the mean value
# data_drop['HandsetPrice'] = data_drop['HandsetPrice'].astype(np.int64)
# data_drop['HandsetPrice'] = data_drop['HandsetPrice'].replace(0, data_drop['HandsetPrice'].mean())
# data_drop['HandsetPrice'].value_counts()

# Age only has 2% Missing for HH1 and HH2 so lets average and replace 

data_drop['AgeHH1'] = data_drop['AgeHH1'].fillna(0)

data_drop['AgeHH1'] = data_drop['AgeHH1'].astype(str).astype(float)
data_drop['AgeHH1'] = data_drop['AgeHH1'] .replace(0, data_drop['AgeHH1'].mean())

data_drop['AgeHH1'] = data_drop['AgeHH1'].astype(int)
data_drop['AgeHH1'].value_counts()

MonthlyRevenue               21
MonthlyMinutes               21
TotalRecurringCharge         21
DirectorAssistedCalls        21
OverageMinutes               21
RoamingCalls                 21
PercChangeMinutes            42
PercChangeRevenues           42

data_drop['PercChangeRevenues'] = data_drop['PercChangeRevenues'].fillna(0)

data_drop['PercChangeRevenues'] = data_drop['PercChangeRevenues'].astype(str).astype(float)
data_drop['PercChangeRevenues'] = data_drop['PercChangeRevenues'] .replace(0, data_drop['AgeHH1'].mean())

data_drop['PercChangeRevenues'] = data_drop['PercChangeRevenues'].astype(int)
data_drop['PercChangeRevenues'].value_counts()

# Replace all 0 in handset price to the mean value
data_drop['AgeHH2'] = data_drop['AgeHH2'].fillna(0)

data_drop['AgeHH2'] = data_drop['AgeHH2'].astype(str).astype(float)
data_drop['AgeHH2'] = data_drop['AgeHH2'] .replace(0, data_drop['AgeHH1'].mean())
data_drop['AgeHH2'] = data_drop['AgeHH2'].astype(int)
data_drop['AgeHH2'].value_counts()

data_drop['AgeHH2'].value_counts()

data_drop.isna().sum()

# Average those ones with missing values as they're minimal

# Replace all 0 in handset price to the mean value (cant do fillna for entire thing doesnt work and cant fill in all 0 so have to do it 1 by 1)

fill_these = ['MonthlyRevenue' ,'MonthlyMinutes','TotalRecurringCharge', 'DirectorAssistedCalls' ,
              'OverageMinutes'  ,'RoamingCalls' ,'PercChangeMinutes', 'PercChangeRevenues' ]

data_drop['MonthlyRevenue'] = data_drop['MonthlyRevenue'].interpolate()
data_drop['MonthlyRevenue'] = data_drop['MonthlyRevenue'].replace(np.nan, data_clean['MonthlyRevenue'].median())
data_drop['MonthlyRevenue'].value_counts()

data_drop_2 = data_drop.copy()

# Replace all 0 in handset price to the mean value
data_drop_2['MonthlyMinutes'] = data_drop_2['MonthlyMinutes'].replace(np.nan, data_drop_2['MonthlyMinutes'].median())
data_drop_2['TotalRecurringCharge'] = data_drop_2['TotalRecurringCharge'].replace(np.nan, data_drop_2['TotalRecurringCharge'].median())
data_drop_2['DirectorAssistedCalls'] = data_drop_2['DirectorAssistedCalls'].replace(np.nan, data_drop_2['DirectorAssistedCalls'].median())

data_drop_2['OverageMinutes'] = data_drop_2['OverageMinutes'].replace(np.nan, data_drop_2['OverageMinutes'].median())
data_drop_2['RoamingCalls'] = data_drop_2['RoamingCalls'].replace(np.nan, data_drop_2['RoamingCalls'].median())
data_drop_2['PercChangeMinutes'] = data_drop_2['PercChangeMinutes'].replace(np.nan, data_drop_2['PercChangeMinutes'].median())
data_drop_2['PercChangeRevenues'] = data_drop_2['PercChangeRevenues'].replace(np.nan, data_drop_2['PercChangeRevenues'].median())

data_drop_2.isna().sum()

# ALL LOOKS CLEAN OMG!!!

data_drop.info

# Move Churn column last (only for the FULL dataset - not used on the uni one)

new_order = [col for col in data_drop.columns if col != 'Churn'] + ['Churn']
cleaned_data = data_drop[new_order]

cleaned_data

# Convert numerical class to nominal for weka
data_nominal = data_clean.copy()
data_nominal['Churn'] = data_nominal['Churn'].replace({1: 'Yes', 0: 'No'})	
data_nominal.head(10)

cleaned_data.to_csv('uni_test_data.csv',mode = 'w', index=False) 
files.download('uni_test_data.csv')

cleaned_data.info