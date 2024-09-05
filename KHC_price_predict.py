#!/usr/bin/env python
# coding: utf-8

# KC'deki ev fiyatlarını tahmin eden bir model geliştiriyoruz.
# veri seti : 

# ## Kütüphaneleri yükleme

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Veri setini yükleme

# In[3]:


df=pd.read_csv('kc_house_data.csv')
df

 Bu veri seti, Seattle'ı içeren King County için ev satış fiyatlarını içermektedir. Mayıs 2014 ile Mayıs 2015 arasında satılan evleri kapsar.

Basit regresyon modellerini değerlendirmek için harika bir veri setidir.

Veri Sözlüğü:
- id: Bir ev için id no
- date: Evin satıldığı tarih
- price: Fiyat, [tahmin hedefi,y]
- bedrooms: Yatak odası sayısı
- bathrooms: Banyo sayısı/yatak odası
- sqft_living: Evdeki alanın metre karecinsinden büyüklüğü
- sqft_lot: Arsanın metre kare cinsinden büyüklüğü
- floors: Evdeki toplam kat sayısı
- waterfront: Su manzaralı ev
- view: Manzara
- condition: Genel olarak evin durumu ne kadar iyi
- grade: King County derecelendirme sistemi temelinde konut birimine verilen genel derece
- sqft_above: Bodrum hariç evin metre kare cinsinden büyüklüğü
- sqft_basement: Bodrumun  metre kare cinsinden büyüklüğü
- yr_built: Yapım yılı
- yr_renovated: Retorasyon yılı
- zipcode: Posta kodu
- lat: Enlem koordinatı
- long: Boylam koordinatı
- sqft_living15: 2015 yılında oturma odası alanı (bazı yenilemeleri ima eder). Bu, arsa büyüklüğü alanını etkilemiş olabilir veya olmayabilir.
- sqft_lot15: 2015 yılında arsa büyüklüğü (bazı yenilemeleri ima eder).
# In[4]:


df.info()


# In[5]:


# linkten csv dosyası yükleme 
# df=pd.read_csv('csv dosyasının linkini buraya yazıyoruz')


# In[6]:


pd.set_option('display.max_columns',None)


# In[7]:


# Otomatik olarak EDA'yı gerçekleştirmek
get_ipython().system('pip install ydata_profiling')
import ydata_profiling


# In[8]:


df.profile_report()
profile=ydata_profiling.ProfileReport(df)
profile.to_file('report.html')


# In[9]:


#EDA ve Veri Ön İşleme


# In[10]:


df[df['price']==df['price'].max()]


# In[11]:


df.describe().T


# In[12]:


# Outlier Kontrolü
# Aykırı değerler 

Normal dağılımda3Sd-+ verilerin %99.6'sını kapsar
Geriye kalan %.4 değerler aykırı değerlerdir. 
# In[13]:


sns.boxplot(x=df['bathrooms'],data=df)


# # Outliers için 1. Yöntem
# df_bedrooms_outliers_min=df2[df2['bedrooms]<df2['bedrooms].mean()-3*df2['bedrooms].std()]

# In[14]:


# 2. yöntem
df_kor=df[['price','bedrooms','bathrooms','sqft_living','sqft_lot']]
outliers=df_kor.quantile(q=.99)
outliers


# In[15]:


df_non_outliers=df[df['price']<outliers['price']]


# In[16]:


df_non_outliers=df_non_outliers[df_non_outliers['bedrooms']<outliers['bedrooms']]


# In[17]:


df_non_outliers=df_non_outliers[df_non_outliers['bathrooms']<outliers['bathrooms']]


# In[18]:


df_non_outliers=df_non_outliers[df_non_outliers['sqft_living']<outliers['sqft_living']]


# In[19]:


df_non_outliers=df_non_outliers[df_non_outliers['sqft_lot']<outliers['sqft_lot']]


# In[20]:


df_non_outliers.describe()


# In[21]:


sns.histplot(df_non_outliers['price'], bins=15, kde=True)


# In[22]:


df_non_outliers['zipcode'].dtype


# In[23]:


df_non_outliers['zipcode']=df_non_outliers['zipcode'].astype('category')


# In[24]:


df_non_outliers['zipcode'].dtype


# In[25]:


df_non_outliers.info()


# In[55]:


plt.figure(figsize=(20,20))
sns.heatmap(df_non_outliers.corr(numeric_only=True), annot=True)


# In[27]:


# Price ile korealsyon sıralaması pozitif olanları alır.
df_cor=df_non_outliers.corr(numeric_only=True).sort_values('price', ascending=False)['price'].head(10)


# In[28]:


df_cor[df_cor>.5]


# In[29]:


# Emlak sektöründeki bilgileri kullanıyoruz.
# Özellik dönüşümü
# yatak odası fiyat üzerinde çok etkili


# In[30]:


df_non_outliers['bedrooms']=df_non_outliers['bedrooms']**2
df_non_outliers['bathrooms']=df_non_outliers['bathrooms']**2
df_non_outliers['sqft_living']=df_non_outliers['sqft_living']**2
df_non_outliers['age']=2015-df_non_outliers.yr_built


# In[31]:


df_non_outliers


# In[32]:


df_non_outliers['age'].max()


# In[33]:


df_non_outliers[df_non_outliers['age']==115]


# In[34]:


df_non_outliers[df_non_outliers['age']==115].iloc[0]


# In[35]:


df_non_outliers['yr_renovated'].describe()


# In[36]:


# restorasyon yapılıpp yapılmadığı bilgisi


# In[37]:


df_non_outliers['yr_renovated']=np.where(df_non_outliers['yr_renovated']==0,0,1)
df_non_outliers['yr_renovated'] # restorasyon yılı sıfırsa sıfır olsun daha fazlaysa 1 olsun


# In[38]:


df_non_outliers['renovated']=df_non_outliers['yr_renovated']
df_non_outliers.drop('yr_renovated',axis=1,inplace=True)


# In[39]:


df_non_outliers.info()


# In[40]:


# bodrum var yok
df_non_outliers['sqft_basement']=np.where(df_non_outliers['sqft_basement']==0,0,1)
df_non_outliers['sqft_basement']


# In[41]:


# Özellikleri ve hedefi belirleyelim
X=df_non_outliers.drop(['price','date','id','lat','long'],axis=1)
y=df_non_outliers['price']


# In[42]:


# Kategorik verileri sayısal verilere dönüştürme
X=pd.get_dummies(X, columns=['zipcode'],drop_first=True)
X


# In[43]:


# Standard Scaler ile Ölçeklendirme
from sklearn.preprocessing import StandardScaler


# In[44]:


scaler=StandardScaler()


# In[45]:


X_scaler=scaler.fit_transform(X)


# In[46]:


# Veri setini eğitim ve test olarak ayırma
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaler,y,test_size=0.2,random_state=42)


# In[47]:


#pip install xgboost


# In[48]:


#pip install lightgbm


# In[49]:


# Modelleme
from all_reg_models import all_reg_models


# In[50]:


all_reg_models(X_train,X_test,y_train,y_test)


# In[51]:


from lightgbm import LGBMRegressor
lgbm_model=LGBMRegressor()


# In[52]:


from sklearn.metrics import r2_score,mean_squared_error
lgbm_model.fit(X_train,y_train)
y_pred=lgbm_model.predict(X_test)
r2=r2_score(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**.5
print('Model in R2 Score:',r2,
     '\nModel in RMSE:',rmse)


# In[53]:


columns=X.columns
df_importance=pd.DataFrame({'Feature':columns,'Importance':lgbm_model.feature_importances_})


# In[54]:


df_importance.sort_values(by='Importance',ascending=False).head(10)


# In[ ]:




