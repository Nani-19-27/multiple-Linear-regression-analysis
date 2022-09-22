#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Analysis - Actual vs Predicted observations
# 
# > About Project -  With the help of this car's data set, I want to know the relationship between the foremost specifications of the vehicle itself. we know Engine cc, BHP, and Torque 
# are more influencing the mileage factor individually. we cannot estimate the mileage figure with engine cc or other specifications(simple regression). however, if we consider the main specifications of the vehicle as an independent variable and mileage as a dependent variable. by doing this analysis, we will be able to estimate the mileage figures
# 

# ## Import DataSet

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')


# ## Reading DataSet and Data Preprocessing 

# In[2]:


df = pd.read_csv("C:\\Users\\Manikanta\\Downloads\\Car details v3.csv")


# In[3]:


df = df[['name','mileage','engine','max_power','torque']] ## choosing columns which are required for analysis.


# In[4]:


df.head() #here name column no need to keep however, at data cleansing part it would be helpfull for cross checking.


# In[5]:


df.info() #rows are 8128 and we can see here data type of each column


# In[6]:


df.isna().sum() #no.of null values per column


# In[7]:


#Note - removing the rows if "all columns" don't have values...

df.dropna(axis='index',how='all',subset=['mileage','engine','max_power','torque'],inplace=True) 


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum() # no.of duplicated values does dataset have


# In[10]:


#note - removing all unnessasary repeated values and and i will remove duplicate again later after remove the name name.

df.drop_duplicates(ignore_index=True,inplace=True)


# In[11]:


#ignore this column go forward... you will understand why did i code, later...

df['torque'].replace({'380Nm(38.7kgm)@ 2500rpm':'380Nm@ 1750-2500rpm'},inplace=True)

#________________________________________________________________________________
df.torque.replace({'190@ 21,800(kgm@ rpm)':'223Nm@'},inplace=True)

df.torque.replace({'145@ 4,100(kgm@ rpm)':'145Nm@'},inplace=True)

df.torque.replace({'110@ 3,000(kgm@ rpm)':'110Nm@'},inplace=True)

df.torque.replace({'130@ 2500(kgm@ rpm)':'130Nm@'},inplace=True)

df.torque.replace({'115@ 2,500(kgm@ rpm)':'115Nm@'},inplace=True)

df.torque.replace({'115@ 2500(kgm@ rpm)':'115Nm@'},inplace=True)
#_______________________________________________________________________________

df.torque.replace({'14.9 KGM at 3000 RPM':'14.9kgm'},inplace=True)

df.torque.replace({'24 KGM at 1900-2750 RPM':'24kgm'},inplace=True)


# In[12]:


#Note - 1. observations with string values are there so step-1 is seperate it and remove str values.
#Note - 2.then convert it into float type from object


df['mileage']  = df['mileage'].str.split().str.get(0).astype(float) 


# In[13]:


#Note - 1. observations with string values are there so step-1 is seperate it and remove str values.
#Note - 2.then convert it into float type from object


df['engine']  = df['engine'].str.split().str.get(0).astype(float)


# In[14]:


#Note - 1. observations with string values are there so step-1 is seperate it and remove str values.

df['max_power'] =df['max_power'].str.split().str.get(0)


# In[15]:


#here i have found a data entry mistake

df[df['max_power']=='bhp']


# In[16]:


#step-1 replace with nan values then....

df.max_power.replace('bhp',np.nan,inplace=True)


# In[17]:


#step-2  converting into float type

df['max_power'] = df['max_power'].astype(float)


# In[18]:


#here we can see how torque observations along with str and special characters.

df.torque.unique()


# In[19]:


#note  1-remoing all str values with observations along with @special character 
#note  2.converting into float type



df['Torque']=df.torque.str.split().str.get(0).str.split('Nm').str.get(0).str.split('Kgm').str.get(0).str.split('kgm').str.get(0).str.split('nm').str.get(0).str.split('NM').str.get(0).str.replace('[\@\,]','').replace({'110(11.2)':'110'}).astype(float)


# In[20]:


df.info() #now it is fine however....


# # ___
# 
# #### here in the torque column there are two types of torque figures are there -
# 
# 1. Nm unit values
# 2. Kg-m unit values
# 
# ### Nm = kgm * 9.8067  for more see the below image
# 
# 
# 
# #### Therefore , we need to convert the values kgm to nm..
# 

# In[21]:


from urllib.request import urlretrieve

urlretrieve('https://th.bing.com/th/id/R.4c321b88fc80b86880bb8662881c13fa?rik=FywKvPuufxs3wA&riu=http%3a%2f%2fwww.unipulse.tokyo%2fen%2fwp-content%2fuploads%2f2018%2f02%2fNm_kgm.gif&ehk=%2bC4wG%2fZtU57DZDu3cqH4XHsclceIyvMOZw564ayZUi4%3d&risl=&pid=ImgRaw&r=0','details.jpg')


# In[22]:


from PIL import Image

torque_det = Image.open('details.jpg')

plt.figure(figsize=(20,10))

plt.grid(False)
plt.axis(False)

plt.imshow(torque_det);


# In[23]:


#  step -1   firsly we have torque named columns are 2 in our data. we should use unmodified column that is "torque"
#            here i will seperate the column, nm and kgm like false and True Boolean type. but first we should know that 
#            kgm string in capital and small letters formate and some data entry mistakes (that is they were mistakenly type kgm
#            instead of nm)  so can check my notework at "In[11]".

add = pd.DataFrame({'bool':df['torque'].str.contains('kgm')})  


# In[24]:


df = pd.concat((df,add),axis=1)

#step -2   concate it. next you can understand. it will be appearing..torque column with nm observation indicating 'False' and
#          torque column with kgm indicating true..


df.head()


# In[25]:


#then divide it..into two types true and false and after calculation we will concate it both.

a = df[df['bool']==True]


# In[26]:


b = df[df['bool']==False]


# In[27]:


#create a column.it should have kgm to nm values

a['n_Torque'] = a.Torque* 9.80665


# In[28]:


#then select required columns

a1 = a[['name','mileage','engine','max_power','n_Torque']]


# In[29]:


#here we will concate them again so keeping columns are same as a1 part

b1 = b.drop(columns=['torque','bool'],axis=True).rename({'Torque':'n_Torque'},axis=1)


# In[30]:


car_df =pd.concat((a1,b1),axis=0,ignore_index=True)


# In[31]:


car_df.reset_index(inplace=True)


# In[32]:


#droping unnesscary columns secound time

car_df.drop(columns=['name','index'],inplace=True)


# In[33]:


#In this analysis we require only figures. so that i had to remove duplicates again

car_df[['mileage','engine','max_power','n_Torque']].drop_duplicates(inplace=True)


# In[34]:


#with the help of describe() function we can see stats values and have you observe the min of mileage ?


car_df.describe()


# In[35]:


#data entry mistake so that i am replacing the value 0 to nan

car_df.mileage.replace(0,np.nan,inplace=True)


# In[36]:


car_df.isnull().sum()


# In[37]:


#ehere why i am fill the null values with help of engine..

car_df.mileage = car_df.groupby('engine')['mileage'].apply(lambda x:x.fillna(x.median()))


# In[38]:


car_df.isnull().sum()


# In[39]:


car_df.dropna(inplace=True)


# ## Skewness and Outliers

# In[40]:


for i in ['engine','max_power','n_Torque','mileage']:
    print(i)
    print('skewness : {:.2f}'.format(car_df[i].skew()))
    print( )
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    car_df[i].hist(grid=False)
    plt.subplot(1,2,2)
    sns.boxplot(x=car_df[i])
    plt.show()


# ## Multivariate Analysis

# In[41]:


plt.figure(figsize=(10,7))

sns.heatmap(car_df.corr(),vmax=1,vmin=-1,annot=True);

#here you can see mileage with every other specifications, a negetive corelation coffiecient.


# ## Observations :
# 
# 1. Here we can see the engine (independent variable) and how it is influencing the mileage (dependent variable) r = -0.61 compare to remain independent variables of torque and max power.
# 
# 2. after the engine variable next max power variable is influencing the mileage r= -0.4
# 
# 3. Along with this, we can see a strong relationship between the engine and torque similarly torque and max power r =0.84
# 
# #### with the help heatmap we could get the "R value" 
# 
# #### lets see regression direction in pair plots
# 
# #  ____

# In[42]:


#wait it will take few more seconds for loading...

sct = sns.pairplot(car_df,kind='kde')

sct.map_lower(sns.kdeplot,levels=4,color='.2');


# ## Linear Regression Model

# ### Define x and y

# In[43]:


x_var = car_df.drop(columns=['mileage'],axis=1).values

y_var = car_df.mileage.values


# ### Spliting the data set into the training set and test set

# In[44]:


x_train,x_test,y_train,y_test = train_test_split(x_var,y_var,test_size=0.20,random_state=0)


# ### Train the model on the Training set

# In[45]:


ml = LinearRegression()

ml.fit(x_train,y_train)


# ### Predict the test set results

# In[46]:


y_pred = ml.predict(x_test)


# ### Evalute the Model

# In[47]:


coff_of_det = r2_score(y_test,y_pred)


print('The coffiecient of Determination is {:.2f}'.format(coff_of_det))
print( )
print('The mean squared error is {:.2f}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
print( )
print('the mean absolute percentage error is {:.2f}'.format(mean_absolute_percentage_error(y_test,y_pred)))


# ### Plot the results

# In[48]:


new_df = pd.DataFrame({'actual_y':y_test,'predicted_y':y_pred,'residuals':y_test-y_pred})


# In[50]:


plt.figure(figsize=(10,10))
plt.title('Correlation between acutal and predicted')

sns.scatterplot(x=new_df.actual_y,y=new_df.predicted_y,palette='deep')

print('the correlation coffiecient r value is {:.2f}'.format(new_df.actual_y.corr(new_df.predicted_y)))


# ### Residual density plot

# In[51]:


plt.figure(figsize=(10,5))

new_df.residuals.plot(kind='kde');


# ### Observations -
# 
# #### here we can see the residual density plot how normally distributed it is. However, this data has outliers and skewed distribution. we could see above. so that our residual plot below the left and right lines went to till (-20,20).
