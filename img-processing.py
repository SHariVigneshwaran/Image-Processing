#!/usr/bin/env python
# coding: utf-8

##Artificial Intelligence for Robotics Mini Project
##Name : Hari Vigneshwaran S
##Roll : 20243015

# ### Importing Libraries

# In[1]:


#importing system libraries
import os
import warnings
warnings.simplefilter('ignore')


# In[2]:


#importing datahandling lib
import numpy as np
import pandas as pd 


# In[3]:


#importing data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#import requirbuff components for image processing
from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[5]:


bill=os.listdir(r'C:\Users\Hari Vigneshwaran\Downloads\ai-project\Dataset\Bill Gates')


# In[6]:


amb=os.listdir(r'C:\Users\Hari Vigneshwaran\Downloads\ai-project\Dataset\Mukesh Ambani')


# In[7]:


buff=os.listdir(r'C:\Users\Hari Vigneshwaran\Downloads\ai-project\Dataset\Warren Buffet')


# ### Reading images as a matrix of numbers

# In[8]:


limit=50
billimages=[None]*limit
j=0

for i in bill:
    if(j<limit):
          billimages[j]=imread(r"C:\Users\Hari Vigneshwaran\Downloads\ai-project\Dataset\Bill Gates/"+i)
          j+=1
    else:
          break


# In[9]:


ambimages=[None]*limit
j=0

for i in amb:
    if(j<limit):
          ambimages[j]=imread(r"C:\Users\Hari Vigneshwaran\Downloads\ai-project\Dataset\Mukesh Ambani/"+i)
          j+=1
    else:
          break


# In[10]:


buffimages=[None]*limit
j=0

for i in buff:
    if(j<limit):
          buffimages[j]=imread(r"C:\Users\Hari Vigneshwaran\Downloads\ai-project\Dataset\Warren Buffet/"+i)
          j+=1
    else:
          break


# In[11]:


imshow(billimages[37])


# In[12]:


imshow(ambimages[36])


# In[13]:


imshow(buffimages[29])


# ### Converting to grayscale

# In[14]:


billimages_gray=[None]*limit
j=3

for i in bill:
    if(j<limit):
        billimages_gray[j]=rgb2gray(billimages[j])
        j+=1
    else:
        break


# In[ ]:


ambimages_gray=[None]*limit
j=0

for i in amb:
    if(j<limit):
        ambimages_gray[j]=rgb2gray(ambimages[j])
        j+=1
    else:
        break


# In[ ]:


buffimages_gray=[None]*limit
j=0

for i in buff:
    if(j<limit):
        buffimages_gray[j]=rgb2gray(buffimages[j])
        j+=1
    else:
        break


# In[ ]:


imshow(billimages_gray[39])


# In[ ]:


imshow(ambimages_gray[45])


# In[ ]:


imshow(buffimages_gray[24])


# In[ ]:


billimages_gray[5].shape


# In[ ]:


ambimages_gray[46].shape


# In[ ]:


buffimages_gray[34].shape


# ### Matrix resize

# In[ ]:


for j in range(50):
    sc=billimages_gray[j]
    billimages_gray[j]=resize(sc,(512,512))


# In[ ]:


for j in range(50):
    amb=ambimages_gray[j]
    ambimages_gray[j]=resize(amb,(512,512))


# In[ ]:


for j in range(50):
    buff=buffimages_gray[j]
    buffimages_gray[j]=resize(buff,(512,512))


# In[ ]:


billimages_gray[4].shape


# In[ ]:


ambimages_gray[4].shape


# In[ ]:


buffimages_gray[46].shape


# In[ ]:


imshow(billimages_gray[12])


# In[ ]:


imshow(ambimages_gray[34])


# In[ ]:


imshow(buffimages_gray[35])


# ### Matrix to vector conversion

# In[ ]:


#find out the no.of grayscale images
len_of_images_sc=len(billimages_gray)


# In[ ]:


len_of_images_sc


# In[ ]:


image_size_sc=billimages_gray[1].shape
image_size_sc


# In[ ]:


flatten_size_sc=image_size_sc[0]*image_size_sc[1]
flatten_size_sc


# In[ ]:


#Flattening the arrays
for i in range(len_of_images_sc):
    billimages_gray[i]=np.ndarray.flatten(billimages_gray[i]).reshape(flatten_size_sc,1)


# In[ ]:


#Stack the individual image array elements into one array
billimages_gray=np.dstack(billimages_gray)


# In[ ]:


#change the axis of the array elements
billimages_gray=np.rollaxis(billimages_gray,axis=2,start=0)


# In[ ]:


billimages_gray.shape


# In[ ]:


billimages_gray=billimages_gray.reshape(len_of_images_sc,flatten_size_sc)
billimages_gray.shape


# ### Create a DataFrame - Bill gates

# In[ ]:


bill_data=pd.DataFrame(billimages_gray)


# In[ ]:


bill_data


# In[ ]:


bill_data['label']="Bill Gates"


# In[ ]:


bill_data


# In[ ]:


len_of_images_amb=len(ambimages_gray)


# In[ ]:


len_of_images_amb


# In[ ]:


image_size_amb=ambimages_gray[1].shape
image_size_amb


# In[ ]:


flatten_size_amb=image_size_amb[0]*image_size_amb[1]
flatten_size_amb


# In[ ]:


for i in range(len_of_images_amb):
    ambimages_gray[i]=np.ndarray.flatten(ambimages_gray[i]).reshape(flatten_size_amb,1)


# In[ ]:


#Stack the individual image array elements into one array
ambimages_gray=np.dstack(ambimages_gray)


# In[ ]:


ambimages_gray=np.rollaxis(ambimages_gray,axis=2,start=0)


# In[ ]:


ambimages_gray.shape


# In[ ]:


ambimages_gray=ambimages_gray.reshape(len_of_images_amb,flatten_size_amb)
ambimages_gray.shape


# ### Creating DataFrame - Mukesh Ambani

# In[ ]:


amb_data=pd.DataFrame(ambimages_gray)
amb_data


# In[ ]:


amb_data['label']="Mukesh Ambani"
amb_data


# In[ ]:


len_of_images_buff=len(buffimages_gray)
len_of_images_buff


# In[ ]:


image_size_buff=buffimages_gray[1].shape
image_size_buff


# In[ ]:


flatten_size_buff=image_size_buff[0]*image_size_buff[1]
flatten_size_buff


# In[ ]:


for i in range(len_of_images_buff):
    buffimages_gray[i]=np.ndarray.flatten(buffimages_gray[i]).reshape(flatten_size_buff,1)


# In[ ]:


buffimages_gray=np.dstack(buffimages_gray)


# In[ ]:


buffimages_gray=np.rollaxis(buffimages_gray,axis=2,start=0)


# In[ ]:


buffimages_gray.shape


# In[ ]:


buffimages_gray=buffimages_gray.reshape(len_of_images_buff,flatten_size_buff)
buffimages_gray.shape


# ### Creating DataFrame - Warren Buffet

# In[ ]:


buff_data=pd.DataFrame(buffimages_gray)
buff_data


# In[ ]:


buff_data['label']="warren buffet"
buff_data


# ## Combining DataFrames

# In[ ]:


half=pd.concat([bill_data,amb_data])


# In[ ]:


total_data=pd.concat([half,buff_data])
total_data


# ### Shuffling rows

# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


data_indexbuff=shuffle(total_data).reset_index()
data_indexbuff


# In[ ]:


data=data_indexbuff.drop(['index'],axis=1)


# In[ ]:


data


# ### Assigning dependent and independent variables

# In[ ]:


x=data.values[:,:-1]
y=data.values[:,-1]
x


# In[ ]:


y


# ### Splitting the dataset

# In[ ]:


from sklearn.ambdel_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ### Import Support Vector Machine

# In[ ]:


from sklearn import svm


# In[ ]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# ### Image prediction

# In[ ]:


y_prbuff=clf.prbuffict(x_test)


# In[ ]:


y_prbuff


# ### Prediction Accuracy

# In[ ]:


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_prbuff)


# In[ ]:


accuracy


# ### Analysis of prbuffiction

# In[ ]:


#Analysis of prbuffiction
from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,y_prbuff)

