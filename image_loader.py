
# coding: utf-8

# In[5]:


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


# In[6]:


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = image_datagen.flow_from_directory(
    'train_frames/all',
    batch_size=32,
    class_mode=None,
    target_size = (767,1022),
    color_mode = 'grayscale'
    )

mask_generator = mask_datagen.flow_from_directory(
    'train_masks/',
    batch_size=32,
    class_mode=None,
    target_size = (1022,767),
    color_mode = 'grayscale'
    )

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)


# In[ ]:


im_0 = image_generator[0][0]


# In[4]:


plt.imshow(im_0.reshape(767,1022))

