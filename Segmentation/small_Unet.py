#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models, metrics
from tensorflow.python.keras import backend as K
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd


# In[2]:


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[3]:


seed = 123
random.seed = seed
np.random.seed = seed
tf.seed = seed


# In[4]:


image_size = 256
batch_size = 10

train_path = 'train/'
test_path = 'validation/'

## Training Ids
train_ids = next(os.walk(os.path.join(train_path + 'images/all/')))[2]
train_ids = [os.path.splitext(x)[0] for x in train_ids]

## Test Ids
test_ids = next(os.walk(os.path.join(test_path + 'images/all/')))[2]
test_ids = [os.path.splitext(x)[0] for x in test_ids]


# # Define own data generator class

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


gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(1)
print(x.shape, y.shape)


# In[7]:


r = random.randint(0, len(x)-1)

fig,ax = plt.subplots(figsize=(8,6))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#ax = fig.add_subplot(1, 2, 1)
#ax.imshow(x[r])
#ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[r], (image_size, image_size)), cmap="gray")
ax.text(20,20,'1',color='red',fontsize=16)
fig.show()


# # Define the convolutional blocks for the U-Net implementation

# In[8]:


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


# In[9]:


def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1]) 
    c3, p3 = down_block(p2, f[2]) 
    c4, p4 = down_block(p3, f[3]) 
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3])
    u2 = up_block(u1, c3, f[2])
    u3 = up_block(u2, c2, f[1])
    u4 = up_block(u3, c1, f[0])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


# In[10]:


def dice_coeff(y_true, y_pred,smooth=0.):
    #smooth = 1.
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


import scipy.spatial.distance


# In[14]:


def jaccard_index(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    #intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    #union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (union - intersection)
    return jac 


# In[15]:


def specificity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


# In[16]:


def sensitivity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(neg_y_true * neg_y_pred)
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity


# In[17]:


model = UNet()
#sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.99, nesterov=True)
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coeff, jaccard_index, 'accuracy', specificity, sensitivity])


# In[126]:


model.load_weights('tmp/Unet-100-0.15.hdf5')


# In[127]:


train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
test_gen = DataGen(test_ids, test_path, image_size=image_size, batch_size=batch_size)


# In[128]:


model.evaluate_generator(test_gen)


# In[ ]:





# In[14]:


save_model_path ="tmp/Unet-{epoch:02d}-{val_dice_loss:.2f}.hdf5"
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)


# In[15]:


train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
test_gen = DataGen(test_ids, test_path, image_size=image_size, batch_size=batch_size)


# In[16]:


train_steps = len(train_ids)//batch_size
test_steps = len(test_ids)//batch_size
epochs = 100


# In[17]:


#history = model.fit_generator(train_gen, 
#                              steps_per_epoch=train_steps,
#                              epochs=epochs,
#                              validation_data=test_gen,
#                              validation_steps=test_steps,
#                              callbacks=[cp])


# In[18]:


hist_df = pd.DataFrame(history.history) 
with open('tmp/trainHistoryDict_Unet', 'w') as file_pi:
    hist_df.to_csv(file_pi)


# In[27]:


data = pd.read_csv('tmp/trainHistoryDict_Unet')
data_np = data.values
best_epoch = np.argmin(data_np[:,2])
print('Dice coefficient ', 1 - data_np[:,4])


# In[33]:


plt.plot(data_np[:,0],data_np[:,2],label='training')
plt.plot(data_np[:,0],data_np[:,4],label='validation')
plt.scatter(best_epoch,data_np[best_epoch,4],marker='x',s=150,c='r',label='best model')
plt.xlabel('Epoch')
plt.ylabel('Dice Loss')
plt.legend()
plt.savefig('figures/Dice_loss.png')


# In[17]:


plt.plot(data_np[:,0],data_np[:,1],label='val loss')
plt.plot(data_np[:,0],data_np[:,3],label='train loss')
plt.legend()


# In[39]:


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


# In[40]:


load_model_and_predict(2,test_gen)


# In[ ]:


## Dataset for prediction
x, y = test_gen.__getitem__(100)
result = model.predict(x)
result = result > 0.5


# In[ ]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")


# In[ ]:


#model.save_weights("UNet_2018_256_ep_2.h5")


# In[ ]:


def train_and_save(max_epoch, model_name, train_path, image_size, start_epoch = 0, test_split = 100, batchsize = 10):
    ## Training Ids
    train_ids = next(os.walk("train_2018/images/"))[2]
    train_ids = [os.path.splitext(x)[0] for x in train_ids]
    
    test_ids = train_ids[:val_data_size]
    train_ids = train_ids[val_data_size:]
    
    model = UNet()
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["binary_crossentropy"])
    
    if start_epoch != 0:
        load_name = ''.join([model_name, str(start_epoch),".hd5"])
        model.load_weights(load_name)
    
    train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
    test_gen = DataGen(test_ids, train_path, image_size=image_size, batch_size=1)
    
    train_steps = len(train_ids)//batch_size
    test_steps = len(test_ids)//batch_size

    for i in np.arange(start_epoch,max_epoch):
        print("Epoch %i/%i" %((i+1),max_epoch))
        save_name = ''.join([model_name,str(i+1),".hd5"])
        model.fit_generator(train_gen, validation_data=test_gen,steps_per_epoch=train_steps,  epochs=1)
        model.save_weights(save_name)


# In[ ]:


train_and_save(41,"UNet_2018_256_ep_","train_2018/",256,start_epoch=40)


# In[ ]:


def jaccard_index(model,train_ids,batch_size = 10,image_size = 256,val_data_size = 100):
    ## Test Ids
    #test_ids = next(os.walk(''.join([test_path, "images/"])))[2]
    test_ids = train_ids[:val_data_size]
    #test_ids = [os.path.splitext(x)[0] for x in test_ids]
    
    test_gen = DataGen(test_ids, train_path, image_size=image_size, batch_size=batch_size)
    jaccard_idx = np.zeros(len(test_ids))
    
    for i in range(len(test_gen)):
        x, y = train_gen.__getitem__(i)
        result = model.predict(x)
        result = result > 0.5
        intersection = np.logical_and(y, result)
        union = np.logical_or(y, result)
        jaccard_idx[i:i+batch_size] = np.sum(intersection,axis=[0,1]) / np.sum(union,axis=[0,1])
    return jaccard_idx


# In[ ]:


def evol_jacc_idx(model_name, max_epoch = 2,batch_size = 100):
    jacc_idxs = np.zeros(max_epoch)
    for i in range(max_epoch):
        load_name = ''.join([model_name, str(i + 1),".hd5"])
        model = UNet()
        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["binary_crossentropy"])
        model.load_weights(load_name)
        jacc_idx = jaccard_index(model,train_ids,batch_size=batch_size)
        jacc_idxs[i] = jacc_idx.mean()
    return jacc_idxs


# In[ ]:


indexes = evol_jacc_idx("UNet_2018_256_ep_",batch_size=100)


# In[ ]:


plt.plot(np.arange(len(indexes)),indexes)
plt.xlabel("Epoch");
plt.ylabel("Jaccard-Index");


# In[ ]:


indexes.max()


# In[ ]:


#np.savetxt("jaccard_Unet.txt",indexes)


# In[ ]:


len(test_gen)


# In[ ]:


indexes


# In[ ]:


model.summary()


# In[ ]:


len(test_gen)


# In[ ]:


plt.imsave("pred_Unet/res_2.png", y[0].astype('uint8').reshape(256,256), cmap=matplotlib.cm.gray)


# In[ ]:





# In[ ]:


result[0].astype('uint8').shape


# In[27]:


len(train_ids)


# In[28]:


len(test_ids)


# In[19]:


model.summary()


# In[ ]:




