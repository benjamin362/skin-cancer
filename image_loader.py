
# coding: utf-8

# In[13]:


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img


# In[25]:


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = image_datagen.flow_from_directory(
    'ISBI2016_ISIC_Part1_Training_Data',
    class_mode=None,)

mask_generator = mask_datagen.flow_from_directory(
    'ISBI2016_ISIC_Part1_Training_GroundTruth',
    class_mode=None,)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

