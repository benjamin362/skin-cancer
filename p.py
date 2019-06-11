import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from keras.layers import Activation, Dense
from keras import backend as K
from keras.layers import Convolution2D, Dense
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
#print("intento cnn mia")
#print("intento cnn1")
print("TODO")
meld=pd.read_csv("mel.csv")
df=pd.read_csv("./traindir/newtrain.csv")
dftrain=pd.read_csv("newsampl.csv")
columns=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF","VASC"]
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(
                                            dataframe=dftrain,###dftrain
                                            directory="./traindir/train",
                                            x_col="image",
                                            y_col=columns,
                                            batch_size=32,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(350,350))

valid_generator=test_datagen.flow_from_dataframe(
                                                 dataframe=df[4500:5500],
                                                 directory="./traindir/train",
                                                 x_col="image",
                                                 y_col=columns,
                                                 batch_size=32,
                                                 seed=42,
                                                 shuffle=True,
                                                 class_mode="other",
                                                 target_size=(350,350))

test_generator=test_datagen.flow_from_dataframe(
                                                dataframe=df[5992:10000],
                                                directory="./traindir/train",
                                                x_col="image",
                                                y_col=columns,
                                                batch_size=50,#32
                                                seed=42,
                                                shuffle=True,
                                                class_mode="other",
                                                target_size=(350,350))

####internet ej
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(350, 350, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
# COMPILE
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
'''###first try with 1000 validation
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
'''
'''###try 2: 1 convolutional layer
cnn1 = Sequential()
cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(500,500,3)))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.2))

cnn1.add(Flatten())

cnn1.add(Dense(128, activation='relu'))
cnn1.add(Dense(7, activation='softmax'))

cnn1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
'''
'''
####try 3 +dif
cnn3 = Sequential()
cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
cnn3.add(MaxPooling2D((2, 2)))
#cnn3.add(Dropout(0.25))

cnn3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn3.add(MaxPooling2D(pool_size=(2, 2)))
#cnn3.add(Dropout(0.25))

cnn3.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#cnn3.add(Dropout(0.4))

cnn3.add(Flatten())

cnn3.add(Dense(128, activation='relu'))
cnn3.add(Dropout(0.5))#.3
cnn3.add(Dense(7, activation='softmax'))

cnn3.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])
'''
'''
####try 4 cnn5 with .5 dropout//
cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
cnn4.add(MaxPooling2D((2, 2)))
#cnn3.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
#cnn3.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
#cnn3.add(Dropout(0.4))

cnn4.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
cnn4.add(Flatten())

cnn4.add(Dense(256, activation='relu'))
cnn4.add(Dropout(0.3))#.3
cnn4.add(Dense(7, activation='softmax'))

cnn4.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])
'''
'''
###try cnn5
####try 5
cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
cnn4.add(MaxPooling2D((2, 2)))
#cnn3.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
#cnn3.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
#cnn3.add(Dropout(0.4))

cnn4.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
cnn4.add(MaxPooling2D(pool_size=(2, 2)))

cnn4.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(Dropout(0.5))#.3
cnn4.add(Dense(7, activation='softmax'))

cnn4.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])
'''
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
####unbalanced
'''
from sklearn.utils import class_weight
class_weight = {0: 1113.,
                1: 6705.,
                2: 514. ,
                3: 327. ,
                4: 1099 ,
                5: 115. ,
                6: 142.
                }
'''
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50), ##use 10
                    #class_weight=class_weight)


test_generator.reset()
pred=model.predict_generator(test_generator,
                             steps=STEP_SIZE_TEST,
                             verbose=1)
'''opcion para guardar
    from keras.models import model_from_json
    from keras.models import load_model
    
    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example
    model_json = model.to_json()
    
    
    with open("model_num.json", "w") as json_file:
    json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("model_num.h5")
'''
model.save_weights('50_epochsinternet.h5')
pred_bool = (pred >0.5)
predictions = pred_bool.astype(int)
print("BBBBBBBBBBBBBBBBBB")
print(type(predictions))
print(predictions.shape)
#predictions=predictions.transpose()
#predictions=predictions.reshape(5984, 7)
results=pd.DataFrame(predictions, columns=columns)
results.to_csv("results.csv",index=False)

'''
results["image"]=test_generator.filenames
ordered_cols=["image"]+columns
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)
'''

'''
    
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]




    
    
    
results["image"]=test_generator.filenames
ordered_cols=["image"]+columns
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)
'''
