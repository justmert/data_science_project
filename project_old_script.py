# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %matplotlib inline

# %%
dataset = pd.read_csv('C:\\Users\\LENOVO\\Downloads\\AmesHousing.csv')
dataset


# %%
dataset=dataset.drop(['Order','PID'],axis=1)#we removed two columns that we do not need inside our dataset


# %%
dataset.head()

# %% [markdown]
# In this dataset, we will try to predict sale price for houses with linear regression. First we look all features, missing values and data types. 

# %%
dataset.info()


# %%
dataset.nunique()#number of unique values for each variable


# %%
dataset.isnull().sum()#to check if there is any missing value

# %%
dataset['Lot Frontage'] = dataset['Lot Frontage'].fillna(dataset['Lot Frontage'].mean())
dataset['Alley'] = dataset['Alley'].fillna(dataset['Alley'].mode()[0])
dataset['Mas Vnr Type'] = dataset['Mas Vnr Type'].fillna(dataset['Mas Vnr Type'].mode()[0])
dataset['Mas Vnr Area'] = dataset['Mas Vnr Area'].fillna(dataset['Mas Vnr Area'].mean())
dataset['Bsmt Qual'] = dataset['Bsmt Qual'].fillna(dataset['Bsmt Qual'].mode()[0])
dataset['Bsmt Cond'] = dataset['Bsmt Cond'].fillna(dataset['Bsmt Cond'].mode()[0])
dataset['Bsmt Exposure'] = dataset['Bsmt Exposure'].fillna(dataset['Bsmt Exposure'].mode()[0])
dataset['BsmtFin Type 1'] = dataset['BsmtFin Type 1'].fillna(dataset['BsmtFin Type 1'].mode()[0])
dataset['BsmtFin SF 1'] = dataset['BsmtFin SF 1'].fillna(dataset['BsmtFin SF 1'].mean())
dataset['BsmtFin Type 2'] = dataset['BsmtFin Type 2'].fillna(dataset['BsmtFin Type 2'].mode()[0])
dataset['BsmtFin SF 2'] = dataset['BsmtFin SF 2'].fillna(dataset['BsmtFin SF 2'].mean())
dataset['Bsmt Unf SF'] = dataset['Bsmt Unf SF'].fillna(dataset['Bsmt Unf SF'].mean())
dataset['Total Bsmt SF'] = dataset['Total Bsmt SF'].fillna(dataset['Total Bsmt SF'].mean())
dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])
dataset['Bsmt Full Bath'] = dataset['Bsmt Full Bath'].fillna(dataset['Bsmt Full Bath'].mean())
dataset['Bsmt Half Bath'] = dataset['Bsmt Half Bath'].fillna(dataset['Bsmt Half Bath'].mean())
dataset['Fireplace Qu'] = dataset['Fireplace Qu'].fillna(dataset['Fireplace Qu'].mode()[0])
dataset['Garage Type'] = dataset['Garage Type'].fillna(dataset['Garage Type'].mode()[0])
dataset['Garage Yr Blt'] = dataset['Garage Yr Blt'].fillna(dataset['Garage Yr Blt'].mean())
dataset['Garage Finish'] = dataset['Garage Finish'].fillna(dataset['Garage Finish'].mode()[0])
dataset['Garage Cars'] = dataset['Garage Cars'].fillna(dataset['Garage Cars'].mean())
dataset['Garage Area'] = dataset['Garage Area'].fillna(dataset['Garage Area'].mean())
dataset['Garage Qual'] = dataset['Garage Qual'].fillna(dataset['Garage Qual'].mode()[0])
dataset['Garage Cond'] = dataset['Garage Cond'].fillna(dataset['Garage Cond'].mode()[0])
dataset['Pool QC'] = dataset['Pool QC'].fillna(dataset['Pool QC'].mode()[0])
dataset['Fence'] = dataset['Fence'].fillna(dataset['Fence'].mode()[0])
dataset['Misc Feature'] = dataset['Misc Feature'].fillna(dataset['Misc Feature'].mode()[0])
dataset.info()


# %%
if(dataset.isnull().sum().sum() != 0):
    print('There is null values')
else:
    print('No null values')


# %%
plt.figure(figsize=(10,5))
plt.hist(dataset['SalePrice'],bins=10,edgecolor='black')
plt.xlabel('Sale Price')
plt.ylabel('Number of Houses')
plt.show()


# %%
correlation = dataset.corr() #to  relationship analysis.
plt.figure(figsize=(20, 10))
sns.heatmap(correlation, vmax=.8,annot=True);

# Here we can see correlation between all variables.
# %%
correlation2=dataset.corr()
most_corr=correlation2.index[abs(correlation2["SalePrice"])>0.5]
plt.figure(figsize=(15,10))
g = sns.heatmap(dataset[most_corr].corr(),annot=True,cmap="coolwarm")

# Here we can see most correlated features with the sale price. These features are;
# %%
correlated_features=['Overall Qual','Gr Liv Area','Garage Cars','Garage Area','Total Bsmt SF','1st Flr SF','Year Built','Full Bath','Year Remod/Add','Garage Yr Blt','Mas Vnr Area']
for feature in correlated_features:
    plt.scatter(dataset[feature],dataset['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.show()

#Here we can see scatter plots for most correlated features with sale price.
# %%
sns.distplot(dataset['SalePrice'])#histogram
#skewness and kurtosis
print("Skewness: %f" % dataset['SalePrice'].skew())
print("Kurtosis: %f" % dataset['SalePrice'].kurt())


# %%
dataset['SalePrice_'] = np.log(dataset['SalePrice'])
sns.distplot(dataset['SalePrice_']);
print("Skewness: %f" % dataset['SalePrice_'].skew())
print("Kurtosis: %f" % dataset['SalePrice_'].kurt())
dataset.drop('SalePrice', axis= 1, inplace=True)


