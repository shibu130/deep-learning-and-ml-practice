from itertools import accumulate
from typing import Sequence

from  tensorflow.keras.datasets import mnist

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np

(train,train_label),(test,test_label)=mnist.load_data()


train=train.reshape(60000,28,28,1)
test=test.reshape(10000,28,28,1)


train=train.astype("float32")/255
test=test.astype("float32")/255



network=Sequential()

network.add(layers.Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3),activation="relu"))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Flatten())

network.add(layers.Dense(64,activation="relu"))
network.add(layers.Dense(10,activation="softmax"))
network.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy" ,metrics=['accuracy'])
network.fit(train,train_label,epochs=5,batch_size=128)



#
#array = train[0].reshape(28,28,1,1).T

# print(array.shape)
#pred=network.predict(array)


# pred=network.predict(train_img[0].reshape(28*28))

#f=np.argmax(pred, axis = 1)

#print("{}".format(f[0]))







