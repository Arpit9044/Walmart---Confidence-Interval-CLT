#!/usr/bin/env python
# coding: utf-8

# In[205]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[206]:


df['Occupation'].unique()


# In[207]:


df=pd.read_csv(r"C:\Users\hp\Downloads\walmart_data.csv")
df


# In[208]:


df.shape


# In[209]:


df.info()


# In[210]:


df.describe(include='all')


# There are more than 5 Lakhs transactions.

# In[211]:


df['User_ID'].nunique()


# There are 5891 Distinct Users.

# In[212]:


df.groupby('Gender')['User_ID'].nunique()


# This data has 4225 Unique Male customers and 1666  Unique Female customers.

# In[213]:


df['Product_ID'].nunique()


# There are 3631 Distinct Products.

# In[214]:


df['Age'].nunique()


# In[215]:


df['Age'].unique()


# - This data is having following age groups:
#     - 0-17  Years
#     - 18-25 Years
#     - 26-35 Years
#     - 36-45 Years
#     - 46-50 Years
#     - 51-55 Years
#     - 55+   Years

# In[216]:


df['Occupation'].unique()


# In[217]:


df['Occupation'].nunique()


# In[218]:


df['City_Category'].unique()


# In[219]:


df['City_Category'].nunique()


# In[220]:


df['Product_Category'].unique()


# In[221]:


df['Product_Category'].nunique()


# 3631 Products belong to 20 Product Categories.

# In[222]:


df[df.duplicated()]


# Neither any null values nor any duplicates present.

# In[224]:


df


# In[225]:


sns.boxplot(data=df,y='Purchase')


# In[226]:


sns.histplot(data=df,x='Purchase',bins=20)
plt.ylabel('Frequency')
plt.show()


# In[229]:


sns.kdeplot(data=df,x='Purchase')
plt.ylabel('Frequency')
plt.show()


# In[230]:


sns.boxplot(data=df,y='Purchase')


# In[231]:


Q1,Median,Q3=np.percentile(df['Purchase'],[25,50,75])
print(Q1,Median,Q3)


# In[232]:


IQR=Q3-Q1
Lower_Whisker=Q1-1.5*IQR
Upper_Whisker=Q3+1.5*IQR
IQR,Lower_Whisker,Upper_Whisker


# In[233]:


df['Purchase'].mean()


# - The mean value of Purchase is 9263 whereas the median is 8047.
# - 50 % of Purchases are between 5823 and 12054.

# In[287]:


df.groupby('Gender')['Purchase'].count()


# In[234]:


df.groupby('Gender')['Purchase'].mean()


# In[235]:


male_purchase_data=df.loc[df['Gender']=='M','Purchase']
male_purchase_data


# In[236]:


male_purchase_data.mean()


# # Now, we will predict the mean Purchase amount per gender for the entire population i.e. 50 million males and 50 million females.

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[331]:


n=50000
male_purchase_sample_mean_list=[]
for i in range(10000):
    male_purchase_sample=np.random.choice(male_purchase_data,size=n)
    male_purchase_sample_mean=male_purchase_sample.mean()
    male_purchase_sample_mean_list.append(male_purchase_sample_mean)
sns.histplot(male_purchase_sample_mean_list)


# In[332]:


male_purchase_sample_mean_list=np.array(male_purchase_sample_mean_list)
male_purchase_sample_mean_list.mean()


# In[333]:


np.percentile(male_purchase_sample_mean_list,[2.5,97.5])


# In[334]:


np.percentile(male_purchase_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount of male will be between $9392.79025-$9481.784657
# - 99% of the times the mean purchase amount of male will be between $9377.9041948-$9496.1040259

# In[335]:


Female_purchase_data=df.loc[df['Gender']=='F','Purchase']
Female_purchase_data


# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[336]:


n=50000
Female_purchase_sample_mean_list=[]
for i in range(10000):
    Female_purchase_sample=np.random.choice(Female_purchase_data,size=n)
    Female_purchase_sample_mean=Female_purchase_sample.mean()
    Female_purchase_sample_mean_list.append(Female_purchase_sample_mean)
sns.histplot(Female_purchase_sample_mean_list)


# In[337]:


Female_purchase_sample_mean_list=np.array(Female_purchase_sample_mean_list)
Female_purchase_sample_mean_list.mean()


# In[338]:


np.percentile(Female_purchase_sample_mean_list,[2.5,97.5])


# In[339]:


np.percentile(Female_purchase_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount of female will be between $ 8691.867355 - $ 8776.947414
# - 99% of the times the mean purchase amount of female will be between $ 8678.8173709 - $ 8789.7425091

# # Now, we will predict the mean Purchase amount for each marital status for the entire population i.e. 50 million males and 50 million females.

# In[340]:


df.groupby('Marital_Status')['Purchase'].mean()


# In[341]:


Singles_purchase_data=df.loc[df['Marital_Status']==0,'Purchase']
Singles_purchase_data


# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[342]:


n=50000
Singles_purchase_sample_mean_list=[]
for i in range(10000):
    Singles_purchase_sample=np.random.choice(Singles_purchase_data,size=n)
    Singles_purchase_sample_mean= Singles_purchase_sample.mean()
    Singles_purchase_sample_mean_list.append(Singles_purchase_sample_mean)
sns.histplot(Singles_purchase_sample_mean_list)


# In[343]:


Singles_purchase_sample_mean_list=np.array(Singles_purchase_sample_mean_list)
Singles_purchase_sample_mean_list.mean()


# In[344]:


np.percentile(Singles_purchase_sample_mean_list,[2.5,97.5])


# In[374]:


np.percentile(Singles_purchase_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount of singles will be between $ 9221.574583 - $ 9309.248363
# - 99% of the times the mean purchase amount of singles will be between $ 9207.3960247 - $ 9321.6851716

# In[346]:


Partnered_purchase_data=df.loc[df['Marital_Status']==1,'Purchase']
Partnered_purchase_data


# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[347]:


n=50000
Partnered_purchase_data_sample_mean_list=[]
for i in range(10000):
    Partnered_purchase_data_sample=np.random.choice(Partnered_purchase_data,size=n)
    Partnered_purchase_data_sample_mean=Partnered_purchase_data_sample.mean()
    Partnered_purchase_data_sample_mean_list.append(Partnered_purchase_data_sample_mean)
sns.histplot(Partnered_purchase_data_sample_mean_list)
    


# In[348]:


Partnered_purchase_data_sample_mean_list=np.array(Partnered_purchase_data_sample_mean_list)
Partnered_purchase_data_sample_mean_list.mean()


# In[349]:


np.percentile(Partnered_purchase_data_sample_mean_list,[2.5,97.5])


# In[350]:


np.percentile(Partnered_purchase_data_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount of female will be between $ 9216.4586015 - $ 9305.0503785
# - 99% of the times the mean purchase amount of female will be between $ 9203.7528586 - $ 9317.9100092

# # Now, we will predict the mean Purchase amount for each age group for the entire population i.e. 50 million males and 50 million females.

# In[351]:


df


# In[352]:


df.groupby('Age')['Purchase'].mean()


# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[353]:


purchase_data_0_17=df.loc[df['Age']=='0-17','Purchase']
n=50000
purchase_data_0_17_sample_mean_list=[]
for i in range(10000):
    purchase_data_0_17_sample=np.random.choice(purchase_data_0_17,size=n)
    purchase_data_0_17_sample_mean=purchase_data_0_17_sample.mean()
    purchase_data_0_17_sample_mean_list.append(purchase_data_0_17_sample_mean)
sns.histplot(purchase_data_0_17_sample_mean_list)


# In[354]:


purchase_data_0_17_sample_mean_list=np.array(purchase_data_0_17_sample_mean_list)
np.percentile(purchase_data_0_17_sample_mean_list,[2.5,97.5])


# In[355]:


np.percentile(purchase_data_0_17_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 0-17  will be between $ 8889.0040195 - $ 8978.04308  
# - 99% of the times the mean purchase amount for 0-17e will be between $ 8876.5937331 - $ 8992.6592373

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[356]:


purchase_data_18_25=df.loc[df['Age']=='18-25','Purchase']
n=50000
purchase_data_18_25_sample_mean_list=[]
for i in range(10000):
    purchase_data_18_25_sample=np.random.choice(purchase_data_18_25,size=n)
    purchase_data_18_25_sample_mean=purchase_data_18_25_sample.mean()
    purchase_data_18_25_sample_mean_list.append(purchase_data_18_25_sample_mean)
sns.histplot(purchase_data_18_25_sample_mean_list)


# In[357]:


purchase_data_18_25_sample_mean_list=np.array(purchase_data_18_25_sample_mean_list)
np.percentile(purchase_data_18_25_sample_mean_list,[2.5,97.5])


# In[358]:


np.percentile(purchase_data_18_25_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 18-25 will be between $ 9126.2141005 - $ 9213.8477335
# - 99% of the times the mean purchase amount for 18-25 will be between $ 9111.9204281 - $ 9227.8488288

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[359]:


purchase_data_26_35=df.loc[df['Age']=='26-35','Purchase']
n=50000
purchase_data_26_35_sample_mean_list=[]
for i in range(10000):
    purchase_data_26_35_sample=np.random.choice(purchase_data_26_35,size=n)
    purchase_data_26_35_sample_mean=purchase_data_26_35_sample.mean()
    purchase_data_26_35_sample_mean_list.append(purchase_data_26_35_sample_mean)
sns.histplot(purchase_data_26_35_sample_mean_list)


# In[360]:


purchase_data_26_35_sample_mean_list=np.array(purchase_data_26_35_sample_mean_list)
np.percentile(purchase_data_26_35_sample_mean_list,[2.5,97.5])


# In[361]:


np.percentile(purchase_data_26_35_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 26-35 will be between $ 9209.76219  - $ 9296.984682
# - 99% of the times the mean purchase amount for 26-35 will be between $ 9196.2080251 - $ 9311.6717602

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[362]:


purchase_data_36_45=df.loc[df['Age']=='36-45','Purchase']
n=50000
purchase_data_36_45_sample_mean_list=[]
for i in range(10000):
    purchase_data_36_45_sample=np.random.choice(purchase_data_36_45,size=n)
    purchase_data_36_45_sample_mean=purchase_data_36_45_sample.mean()
    purchase_data_36_45_sample_mean_list.append(purchase_data_36_45_sample_mean)
sns.histplot(purchase_data_36_45_sample_mean_list)


# In[363]:


purchase_data_36_45_sample_mean_list=np.array(purchase_data_36_45_sample_mean_list)
np.percentile(purchase_data_36_45_sample_mean_list,[2.5,97.5])


# In[364]:


np.percentile(purchase_data_36_45_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 36-45 will be between $ 9286.072378 - $ 9374.93561 
# - 99% of the times the mean purchase amount for 36-45 will be between $ 9273.2268646 - $ 9388.01571  

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[365]:


purchase_data_46_50=df.loc[df['Age']=='46-50','Purchase']
n=50000
purchase_data_46_50_sample_mean_list=[]
for i in range(10000):
    purchase_data_46_50_sample=np.random.choice(purchase_data_46_50,size=n)
    purchase_data_46_50_sample_mean=purchase_data_46_50_sample.mean()
    purchase_data_46_50_sample_mean_list.append(purchase_data_46_50_sample_mean)
sns.histplot(purchase_data_46_50_sample_mean_list)


# In[366]:


purchase_data_46_50_sample_mean_list=np.array(purchase_data_46_50_sample_mean_list)
np.percentile(purchase_data_46_50_sample_mean_list,[2.5,97.5])


# In[367]:


np.percentile(purchase_data_46_50_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 46-50 will be between $ 9163.6550735 - $ 9251.195744 
# - 99% of the times the mean purchase amount for 46-50 will be between $ 9150.7667831 - $ 9264.2992001

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[368]:


purchase_data_51_55=df.loc[df['Age']=='51-55','Purchase']
n=50000
purchase_data_51_55_sample_mean_list=[]
for i in range(10000):
    purchase_data_51_55_sample=np.random.choice(purchase_data_51_55,size=n)
    purchase_data_51_55_sample_mean=purchase_data_51_55_sample.mean()
    purchase_data_51_55_sample_mean_list.append(purchase_data_51_55_sample_mean)
sns.histplot(purchase_data_51_55_sample_mean_list)


# In[369]:


purchase_data_51_55_sample_mean_list=np.array(purchase_data_51_55_sample_mean_list)
np.percentile(purchase_data_51_55_sample_mean_list,[2.5,97.5])


# In[370]:


np.percentile(purchase_data_51_55_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 51-55 will be between $ 9490.5286875 - $ 9578.8216815
# - 99% of the times the mean purchase amount for 51-55 will be between $ 9473.98568   - $ 9594.8853393

# - We are taking sample size of n i.e. 50k and doing the experiment 10k times.

# In[371]:


purchase_data_55_plus=df.loc[df['Age']=='55+','Purchase']
n=50000
purchase_data_55_plus_sample_mean_list=[]
for i in range(10000):
    purchase_data_55_plus_sample=np.random.choice(purchase_data_55_plus,size=n)
    purchase_data_55_plus_sample_mean=purchase_data_55_plus_sample.mean()
    purchase_data_55_plus_sample_mean_list.append(purchase_data_55_plus_sample_mean)
sns.histplot(purchase_data_55_plus_sample_mean_list)


# In[372]:


purchase_data_55_plus_sample_mean_list=np.array(purchase_data_55_plus_sample_mean_list)
np.percentile(purchase_data_51_55_sample_mean_list,[2.5,97.5])


# In[373]:


np.percentile(purchase_data_55_plus_sample_mean_list,[0.5,99.5])


# - 95% of the times the mean purchase amount for 55+ will be between $ 9490.5286875 - $ 9578.8216815
# - 99% of the times the mean purchase amount for 55+ will be between $ 9278.7901061 - $ 9392.6585926
