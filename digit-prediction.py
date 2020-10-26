from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
from sys import  exit
(train_img,train_label),(test_img,test_label)=mnist.load_data()
# just checking the image

# plt.imshow(train_img[0])   
# plt.show()
# exit()


#scaling the pixels and converting int to float32 , if integers were used then there would be big numbers and it would slow up network 
# /255 is used to make sure that the pixels whose values are between 0-255  are between 0-1 

train_img=train_img.reshape(60000,28*28)
train_img=train_img.astype('float32')/255
test_img=test_img.reshape(10000,28*28)
test_img=test_img.astype('float32')/255

network=models.Sequential()

network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))

network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

network.fit(train_img,train_label,epochs=5,batch_size=128)

test_loss, test_acc=network.evaluate(test_img,test_label)

#
array = train_img[0].reshape(784,1).T

# print(array.shape)
pred=network.predict(array)


# pred=network.predict(train_img[0].reshape(28*28))

f=np.argmax(pred, axis = 1)

print("{}".format(f[0]))



