import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator

# Augment and normalize image data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   zca_whitening = True,
                                   horizontal_flip = True,
                                   rotation_range = 30)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_dir = 'C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Tensorflow_test_pneumonia\\Input\\train'
val_dir = 'C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Tensorflow_test_pneumonia\\Input\\val'
test_dir = 'C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Tensorflow_test_pneumonia\\Input\\test'

"""Specify directories of training, testing, and validation data,
resize all images found to 100x100, and classify as binary classification"""
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (100, 100),
    batch_size = 32,
    class_mode = "binary",
    color_mode = "rgb",
    shuffle = True)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (100, 100),
    batch_size = 16,
    class_mode = "binary",
    color_mode = "rgb",
    shuffle = True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (100, 100),
    batch_size = 1,
    class_mode = "binary",
    color_mode = "rgb",
    shuffle = False)

# Define sequential model and add convolution and max pooling layers
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

# Flatten into one-dimensional array and add Dropout layer
model.add(Flatten())
model.add(Dropout(0.75))

model.add(Dense(512)) # Add first non-linear layer
model.add(BatchNormalization()) # Normalize input data
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid')) # Add final linear layer

# Compile model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

epochs = 10

# Train the model and train against validation dataset
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = 10)

# Evaluate the model on test dataset
test_eval = model.evaluate_generator(test_generator,
                                     steps = 25,
                                     verbose = 1)

# Print accuracy and loss metrics for test_eval
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i] + ": " + str(test_eval[i])))

# Retrieve a list of accuracy results on training and test data sets
# for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data sets for
# each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.legend()

plt.show()