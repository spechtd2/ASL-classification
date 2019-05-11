import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy

batch_size = 16
num_classes = 29

# input image dimensions
img_rows, img_cols, ch = 100, 100, 3
img_rows, img_cols, ch = 100, 100, 3
input_shape = (img_rows, img_cols, ch)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #width_shift_range = 0.1,
        #height_shift_range = 0.1,
        horizontal_flip=False,
        validation_split=0.1)

#############################################################################
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'asl_alphabet_train',  # this is the target directory
        target_size=(100, 100),  # all images will be resized to
        batch_size=batch_size,
		subset='training')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = train_datagen.flow_from_directory(
        'asl_alphabet_train',
        target_size=(100, 100),
        batch_size=batch_size,
		subset='validation')

#############################################################################3
"""
from keras.applications.resnet50 import ResNet50, preprocess_input
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = Flatten()(x)

x = (Dense(256, activation='relu')(x))
#x = (Dropout(0.95)(x))
x = (Dense(128, activation='relu')(x))
x = (Dense(num_classes, activation='softmax')(x))
model = Model(inputs=base_model.input, outputs=x)
"""
model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
################################################################################
model.summary	
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()			  
history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=400 // batch_size)
model.save_weights('Another_Try.h5')  # always save your weights after training or during training
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()