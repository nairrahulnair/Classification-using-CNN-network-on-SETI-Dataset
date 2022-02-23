import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import array_to_img
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

classes=['squiggle', 'narrowband', 'narrowbanddrd', 'noise']

df_xtrain=pd.read_csv('D:/NIT Assignments/DL/DL test 1/train/images.csv', header=None)
y_train=pd.read_csv('D:/NIT Assignments/DL/DL test 1/train/labels.csv', header=None)
y_train.columns=[classes]
df_xtest=pd.read_csv('D:/NIT Assignments/DL/DL test 1/valid/images1.csv', header=None)
y_test=pd.read_csv('D:/NIT Assignments/DL/DL test 1/valid/labels1.csv', header=None)
#df_train=np.array(df_train)

y_train
y_test.columns=[classes]
y_test
df_xtrain.shape

#### x_train-resizing the dataset to 3200 x images, size 64 x 128, channel x 1
#### x-test - resizing it to 800 images with rest remaining the same

df_xtrain=df_xtrain.values.reshape(3200,64,128,1)   
df_xtest=df_xtest.values.reshape(800,64,128,1)


df_xtrain
###Visualizing a particular image
image=df_xtrain[132, :]
plt.imshow(image, cmap='gray')


#### Data generator for better training

gen_train=ImageDataGenerator(horizontal_flip=True)
gen_train.fit(df_xtrain)
gen_test=ImageDataGenerator(horizontal_flip=True)
gen_test.fit(df_xtest)

df_xtrain.shape

model=Sequential()

model.add(Conv2D(32, kernel_size=3, activation='tanh',input_shape=(64,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, kernel_size=3, activation='tanh'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(20, activation='tanh'))
model.add(Dense(4, activation='softmax'))



model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(df_xtrain, y_train, batch_size=64, epochs=8, validation_data=(df_xtest, y_test))

history2=model.fit(df_xtrain, y_train, batch_size=64, epochs=8, validation_data=(df_xtest, y_test))


#### Model Accuracy

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.legend(['train', 'test'], loc='upper right')
plt.show()



history2.history.keys()

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy II')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.legend(['train', 'test'], loc='upper right')
plt.show()

############# SUMMARY
## I ran many tests as per the given sheet and I can tell History 2 is a better
## fit and produces better result than the previous one.

