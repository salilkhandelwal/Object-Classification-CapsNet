from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from layers import CoupledConvCapsule,CapsMaxPool,CapsuleNorm,Length,CapsuleLayer#,#margin_loss,seg_margin_loss
from losses import margin_loss, seg_margin_loss
from keras import optimizers
from keras import metrics
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# dimensions of our images
img_width, img_height = 32, 32
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
print(x_test.shape)
print(y_test.shape)
print(y_test)
datagen.fit(x_test)
#load the model we saved
#model = load_model('model9_try.h5')
model = load_model('model9_try.h5', custom_objects={
    'CoupledConvCapsule': CoupledConvCapsule,
    'CapsMaxPool': CapsMaxPool,
    'CapsuleNorm': CapsuleNorm,
    'CapsuleLayer': CapsuleLayer,
    'Length': Length,
    '_margin_loss': margin_loss(
        downweight=0.5, pos_margin=0.7, neg_margin=0.3
    ),
    '_seg_margin_loss': seg_margin_loss(),
})

model.summary()



correct_classes =0
on_numb_of_images =0
for X_batch, y_batch in datagen.flow(x_test, y_test, batch_size=1,shuffle=False):
  
  
  on_numb_of_images = on_numb_of_images + 1 # only predict few sub set of images 
  classes = model.predict(X_batch,verbose=0,steps=None)
  index = np.argmax(classes)
  if (index == y_batch[0][0]):
      correct_classes= correct_classes +1

  if(on_numb_of_images == 500 ): # remove the code , if all the y_test needed
    break


print("Accuracy is :", (correct_classes/on_numb_of_images)*100)