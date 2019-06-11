#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import zipfile
import functools
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import random


# In[2]:


import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K


# In[3]:


seed = 123
random.seed = seed
np.random.seed = seed
tf.seed = seed


# In[4]:


image_size = 256
batch_size = 5

train_path = 'train/'
test_path = 'validation/'

## Training Ids
train_ids = next(os.walk(os.path.join(train_path + 'images/all/')))[2]
train_ids = [os.path.splitext(x)[0] for x in train_ids]

## Test Ids
test_ids = next(os.walk(os.path.join(test_path + 'images/all/')))[2]
test_ids = [os.path.splitext(x)[0] for x in test_ids]


# In[5]:


class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=5, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, "images/all/", id_name + ".jpg")
        mask_path = os.path.join(self.path, "masks/all/", id_name + ".png")
        
        ## Reading Image
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = cv2.imread(mask_path, -1)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = np.expand_dims(mask, axis=-1)
            
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))


# In[6]:


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder


# In[7]:


train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
test_gen = DataGen(test_ids, test_path, image_size=image_size, batch_size=batch_size)


# In[8]:


inputs = layers.Input(shape=(image_size, image_size, 3))
# 256

encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 128

encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 64

encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 32

encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 16

encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
# 8

center = conv_block(encoder4_pool, 1024)
# center

decoder4 = decoder_block(center, encoder4, 512)
# 16

decoder3 = decoder_block(decoder4, encoder3, 256)
# 32

decoder2 = decoder_block(decoder3, encoder2, 128)
# 64

decoder1 = decoder_block(decoder2, encoder1, 64)
# 128

decoder0 = decoder_block(decoder1, encoder0, 32)
# 256

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)


# In[9]:


model = models.Model(inputs=[inputs], outputs=[outputs])


# In[10]:


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


# In[11]:


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


# In[12]:


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# In[13]:


def jaccard_index(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    #intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    #union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (union - intersection)
    return jac


# In[14]:


def specificity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


# In[15]:


def sensitivity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(neg_y_true * neg_y_pred)
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity


# In[20]:


model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, dice_coeff, jaccard_index, 'accuracy', specificity, sensitivity])


# In[21]:


save_model_path ="tmp/weights-CNN-_test_{epoch:02d}-{val_dice_loss:.2f}.hdf5"
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)


# In[22]:


train_steps = len(train_ids)//batch_size
test_steps = len(test_ids)//batch_size
epochs = 1


# In[23]:


history = model.fit_generator(train_gen, 
                              steps_per_epoch=train_steps,
                              epochs=epochs,
                              validation_data=test_gen,
                              validation_steps=test_steps,
                              callbacks=[cp])


# In[ ]:


hist_df = pd.DataFrame(history.history) 
with open('trainHistoryDict_CNN', 'w') as file_pi:
    hist_df.to_csv(file_pi)


# In[14]:


model.load_weights('tmp/weights-CNN-85-0.13.hdf5')


# In[26]:


def jaccard_index(model,train_ids,batch_size = 1,image_size = 256,val_data_size = 100):
    ## Test Ids
    #test_ids = next(os.walk(''.join([test_path, "images/"])))[2]
    test_ids = train_ids[:val_data_size]
    #test_ids = [os.path.splitext(x)[0] for x in test_ids]
    
    test_gen = DataGen(test_ids, train_path, image_size=image_size, batch_size=batch_size)
    jaccard_idx = tf.zeros([len])
    
    for i in range(len(test_ids)):
        x, y = train_gen.__getitem__(i)
        result = model.predict(x)
        result = result > 0.5
        intersection = np.logical_and(y, result)
        union = np.logical_or(y, result)
        jaccard_idx[i] = np.sum(intersection) / np.sum(union)
    return jaccard_idx


# In[40]:


worst_idx = np.argmin(jaccard_idx)
best_idx = np.argmax(jaccard_idx)


# In[75]:


model.load_weights('weights-CNN-02-0.33.hdf5')


# In[76]:


## Dataset for prediction
x, y = test_gen.__getitem__(1)
result = model.predict(x)
result = result > 0.5


# In[77]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")


# In[39]:


## Dataset for prediction
x, y = train_gen.__getitem__(0)
result = model.predict(x)
result = result > 0.5


# In[40]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")


# In[57]:


def jaccard_coeff(model,y_true,x_true):
    y_pred = model.predict(x_true)
    x_true_f = tf.reshape(x_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    result = y_pred_f > 0.5
    intersection = tf.math.logical_and(y, result)
    intersection = tf.dtypes.cast(intersection,tf.int8)
    union = tf.math.logical_or(y, result)
    union = tf.dtypes.cast(union,tf.int8)
    jaccard_score = tf.math.reduce_sum(intersection) / tf.math.reduce_sum(union)
    return jaccard_score.eval()


# In[58]:


t = jaccard_coeff(model,y,x)


# In[48]:


t


# In[56]:


a = cv2.imread('validation/images/all/ISIC_0010002.jpg',1)


# In[63]:


a.shape


# In[ ]:


def load_model_and_predict(max_epoch,test_gen):
    for j in range(max_epoch):
        model = UNet()
        model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
        model_path = os.path.join("UNet_2018_256_ep_" + str(j + 1) + ".hd5")
        model.load_weights(model_path)
        
        for i in range(len(test_gen)):
            # predict the mask for each picture
            x, y = test_gen.__getitem__(i)
            result = model.predict(x)
            result = result > 0.5
            result = result[0].astype('uint8').reshape(256,256)
            fig,ax = plt.subplots(figsize=(8,6))
            ax.imshow(result, cmap="gray")
            ax.text(20,20,str(j + 1),color='red',fontsize=16)
            fig.savefig(os.path.join("pred_Unet/pred_" + str(i) + "_epoch_" + str(j + 1).zfill(2) + ".png"))
            plt.close()
            if j == 0:
                plt.imsave(os.path.join("pred_Unet/mask_" + str(i) +".png"),y[0].astype('uint8').reshape(256,256), cmap='gray')


# In[43]:


from glob import glob
from os import getcwd, chdir
 
def list_files3(directory, extension):
    saved = getcwd()
    chdir(directory)
    it = glob('weights*')
    chdir(saved)
    it = np.asarray(it)
    it = np.sort(it)    
    return it


# In[44]:


list_files3('tmp/', 'weights')


# In[146]:


def load_evaluate(directory='tmp/'):
    weight_files = list_files3(directory, 'weights')
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, dice_coeff, jaccard_index, 'accuracy', specificity, sensitivity])
    test_steps = len(test_ids)//batch_size
    test_gen = DataGen(test_ids, test_path, image_size=image_size, batch_size=batch_size)
    dict = {0:1,1:2,2:4,3:9,4:13,5:19,6:27,7:29,8:36,9:36,10:45,11:52,12:57,13:62,14:65,15:66,16:69,17:79,18:83,19:85,19:86,20:88,21:90,22:91}
    evaluations = np.zeros([len(weight_files),7])
    for idx in range(len(weight_files)):
        weights = os.path.join('tmp/', weight_files[idx])
        
        model.load_weights(weights)
        evaluations[idx,:] = model.evaluate_generator(test_gen)
        
        epoch = dict[idx]
        make_plot(model,epoch)
    return evaluations


# In[147]:


evaluations = load_evaluate()


# In[208]:


dict_ = {0:1,1:2,2:4,3:9,4:13,5:19,6:27,7:29,8:36,9:36,10:45,11:52,12:57,13:62,14:65,15:66,16:69,17:79,18:83,19:85,19:86,20:88,21:90,22:91}
indices = np.array(list(dict_.values()))
eval_df = pd.DataFrame(np.hstack([indices[:,np.newaxis],evaluations]),index = indices,columns=['Epoch', 'Loss' ,'Dice_Loss', 'Dice_Coeff', 'Jaccard_Index', 'Accuracy', 'Specificity', 'Sensitivity'])
eval_df.reset_index()
with open('tmp/evaluation_CNN', 'w') as file_pi:
    eval_df.to_csv(file_pi)


# In[213]:


data_np = eval_df.values


# In[212]:


plt.plot(eval_df['Epoch'],eval_df['Loss'],label='Validatoon Loss')
plt.plot(eval_df['Epoch'],eval_df['Accuracy'],label='Validaton Accuracy')
plt.ylabel('Dice Loss')
plt.legend()
plt.savefig('figures/Dice_loss_CNN.png')


# In[219]:


eval_df['index'==19]


# In[ ]:





# In[ ]:





# In[72]:


model = models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, dice_coeff, jaccard_index, 'accuracy', specificity, sensitivity])

model.load_weights('tmp/weights-CNN-01-0.34.hdf5')
model.evaluate_generator(test_gen)


# In[ ]:


[dice_loss, dice_coeff, jaccard_index, 'accuracy', specificity, sensitivity]


# In[51]:


[1.0592929778721154,
 0.3939694,
 0.60603094,
 0.4740357,
 0.8598573,
 0.92613745,
 0.85138637]


# In[140]:


def make_plot(model,epoch): 
    x, y = test_gen.__getitem__(87)
    result = model.predict(x)
    result = result > 0.5
    result = result[0].astype('uint8').reshape(256,256)
    fig = plt.figure(figsize=(18, 16))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.asarray(y.reshape(256,256)*255.,dtype=int),cmap="gray")
    ax.text(20,20,str(epoch),color='red',fontsize=20)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.asarray(result.reshape(256,256)*255.,dtype=int),cmap="gray")


    fig.savefig(os.path.join("figures/pred_epoch_" + str(epoch).zfill(2) + ".png"))
    plt.close()


# In[111]:


x.shape


# In[187]:


np.vstack([indices,evaluations])


# In[205]:


np.hstack([indices[:,np.newaxis],evaluations])


# In[200]:


indices.shape


# In[201]:


evaluations.shape


# In[ ]:




