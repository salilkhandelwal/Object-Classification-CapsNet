from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from layers import CoupledConvCapsule,CapsMaxPool,CapsuleNorm,Length,CapsuleLayer#,#margin_loss,seg_margin_loss
from losses import margin_loss, seg_margin_loss
from keras import optimizers
from keras import metrics
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
# load the model we saved
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

pred_arr =np.zeros(1)

for image in x_test:
	x = np.expand_dims(image, axis=0)
	classes = model.predict(x,verbose=0,steps=None)
	prediction = np.argmax(classes)
	pred_arr = np.append(pred_arr, prediction)

pred_arr = np.delete(pred_arr,0)
pred_fin = np.reshape(pred_arr,(10000,1))

count = 0

for x,y in zip(np.nditer(pred_fin), np.nditer(y_test)):
	if x == y:
		count = count + 1

accuracy = count/10000
print(accuracy)