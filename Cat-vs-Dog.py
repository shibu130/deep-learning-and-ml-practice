from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import  RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#model.summary()


#compiling model
model.compile(loss="binary_crossentropy",optimizer=RMSprop(lr=1e-4),metrics=["acc"])

#image preprocessing# train_generatpr=train_datagen.flow_from_directory(
#
# )


train_datagen=ImageDataGenerator(rescale=1./255
                                 ,rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2
                                 ,shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True

                                 )
test_datagen=ImageDataGenerator(rescale=1./255)
validation_datagen=ImageDataGenerator(rescale=1./255)

train_dir=os.path.join("train")
test_dir=os.path.join("test")
valid_dir=os.path.join("valid")


train_generator=train_datagen.flow_from_directory(
train_dir,target_size=(150,150),batch_size=20,class_mode='binary'
)

valid_generator=validation_datagen.flow_from_directory(
valid_dir,target_size=(150,150),batch_size=20,class_mode='binary'
)

test_generator=test_datagen.flow_from_directory(
    test_dir,target_size=(150,150),batch_size=20,class_mode='binary'
)

history=model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=valid_generator,
    validation_steps=50)

model.save('cat_vs_dog.h5-aug')


#prediction remaining










