
# coding: utf-8

# In[633]:

import pandas as pd
import missingno as msno
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing


get_ipython().magic(u'matplotlib inline')




# In[634]:

train_df=pd.read_csv('Sberbank-Russian-Housing-Market/train.csv',parse_dates=['timestamp']).set_index('id')
test_df=pd.read_csv('Sberbank-Russian-Housing-Market/test.csv',parse_dates=['timestamp']).set_index('id')
train_df['isTrain']=1
test_df['isTrain']=0
combine_df=pd.concat([train_df,test_df])
combine_df.shape
macro_df=pd.read_csv('Sberbank-Russian-Housing-Market/macro.csv',parse_dates=['timestamp'])


# In[635]:

print(combine_df.shape)


# In[636]:

combine_df=pd.merge(combine_df,macro_df,on='timestamp',how='left')
combine_df.head()


# In[637]:

print combine_df.timestamp.dtype


# In[638]:

print(combine_df.shape)
print(macro_df.shape)
print(train_df.shape)
print(test_df.shape)


# In[639]:

col_missing = train_df.isnull().any()[train_df.isnull().any()].index
#msno.matrix(df=train_df[col_missing], figsize=(20, 14), color=(0.42, 0.1, 0.05))
train_df[col_missing].apply(lambda c:len(c[c.isnull()])).sort_values()
msno.heatmap(df=train_df[col_missing])


# In[640]:

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns=['column_name','missing_count']
missing_df=missing_df[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[641]:

#dealing with outliers


# In[642]:

plt.scatter(combine_df.full_sq,combine_df.life_sq,s=4,c='red')


# In[643]:

combine_df.loc[combine_df.full_sq>1000,'full_sq']=np.nan
combine_df.loc[combine_df.life_sq>1000,'life_sq']=np.nan
combine_df.loc[combine_df.life_sq>combine_df.full_sq*0.8,'life_sq']=np.nan
combine_df.loc[combine_df.full_sq<3,'full_sq'] = np.nan
combine_df.loc[combine_df.life_sq<3,'life_sq']=np.nan
plt.scatter(combine_df.full_sq,combine_df.life_sq,s=4,c='red')


# In[644]:

plt.scatter(combine_df.kitch_sq,combine_df.full_sq,s=4)
combine_df.loc[combine_df.kitch_sq>combine_df.life_sq]=np.nan
combine_df.loc[combine_df.kitch_sq>500,'kitch_sq']=np.nan
combine_df.loc[combine_df.kitch_sq<2,'kitch_sq']=np.nan



# In[645]:


sns.distplot(train_df.loc[train_df['price_doc']>np.log1p(20000000),'price_doc'])




# In[646]:

plt.scatter(combine_df.build_year,combine_df.price_doc,s=4,c='green')
combine_df.loc[combine_df.build_year>2018,'build_year']
combine_df.loc[combine_df.build_year==20052009.0,'build_year']=2005
combine_df.loc[combine_df.build_year==4965,'build_year'] = np.nan
combine_df.loc[combine_df.build_year>2021,'build_year'] = np.nan
combine_df.build_year.describe(percentiles= [0.9999,0.10])
combine_df.loc[combine_df.build_year<1950,'build_year']=np.nan






# In[647]:

plt.scatter(combine_df.state,combine_df.build_year,s=4)
combine_df.loc[combine_df.state>30,'state']=np.nan




# In[648]:

plt.scatter(combine_df.num_room,combine_df.max_floor,s=4)
combine_df.loc[combine_df.num_room>15,'num_room']=np.nan
combine_df.loc[combine_df.num_room==0,'num_room']=np.nan
combine_df.loc[combine_df.max_floor==0]=np.nan


# In[649]:

plt.scatter(combine_df.max_floor,combine_df.floor,s=4)
combine_df.loc[combine_df.max_floor<combine_df.floor,'max_floor']=np.nan



# In[650]:

month_year = (combine_df.timestamp.dt.month + combine_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
month_year_cnt_map
combine_df['month_year_cnt'] = month_year.map(month_year_cnt_map)
week_year=combine_df['timestamp'].dt.weekofyear+combine_df['timestamp'].dt.year*100
week_year_cnt_map=week_year.value_counts().to_dict()
combine_df['week_year_cnt_map']=week_year.map(week_year_cnt_map)
combine_df['month']=combine_df['timestamp'].dt.month
combine_df['dayofweek']=combine_df['timestamp'].dt.dayofweek
combine_df.drop('timestamp',axis=1,inplace=True)


# In[651]:

combine_df[['month_year_cnt','dayofweek']]


# In[652]:

combine_df.shape


# In[ ]:




# In[653]:

train_df.full_sq.describe()


# In[ ]:




# In[655]:

combine_df['area_km']=combine_df['area_m']/1000000
combine_df['desity']=combine_df['raion_popul']/combine_df['area_km']


# In[656]:

for c in combine_df.columns:
    if combine_df[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(combine_df[c].values))
        combine_df[c] = lbl.transform(list(combine_df[c].values))
        # x_train.drop(c,axis=1,inplace=True)


# In[657]:

print combine_df.shape


# In[658]:

cor=train_df.corr()
price_corr=cor.price_doc


# In[ ]:




# In[659]:

colormap = plt.cm.magma
plt.figure(figsize=(32,24))
plt.title(u'Pearson_bin_other', y=1.05, size=15)
sns.heatmap(train_df.corr())




# In[ ]:





# In[660]:

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()


# In[665]:

combine_df['market_shop']=1/combine_df['market_shop_km']
combine_df['old']=2018-combine_df['build_year']


# In[667]:

sns.clustermap(combine_df[['area_m' ,'desity','raion_popul','sport_objects_raion','park_km','big_market_km','market_shop_km','market_shop','old',
                  'price_doc']].corr(),annot=True)

plt.show()


# In[668]:

train_df=combine_df.iloc[:30471,:]
test_df=combine_df.iloc[30471:,:]
train_df['mean_price']=train_df['price_doc']/train_df['full_sq']
train_df.loc[train_df.mean_price>500000,'mean_price']=np.nan
plt.scatter(train_df.mean_price,train_df.full_sq,s=4)
outputFile = 'train_featured.csv'
train_df.to_csv(outputFile,index=False)
outputFile = 'test_featured.csv'
test_df.to_csv(outputFile,index=False)


# In[ ]:



