import keras.datasets
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
(X_train, y), (X_test, y_test_fine) = keras.datasets.cifar100.load_data(label_mode='fine')

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(X_train)
for X_batch, y_batch in datagen.flow(X_train, y, shuffle=False):
  print(X_batch[0])
  break