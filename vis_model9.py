from argparse import ArgumentParser

parser = ArgumentParser(description='visualize model 9 activations')

parser.add_argument('--model_file', type=str, required=True, help='Model 9 file to load')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar100', help='Dataset to use')
parser.add_argument('--type', type=str, choices=['test', 'train'], default='train', help='Data type to visualize against')

args = parser.parse_args()

from keras.models import load_model
from keract import get_activations, display_activations
import tensorflow as tf
from layers import CoupledConvCapsule, CapsMaxPool, CapsuleNorm, CapsuleLayer, Length
from losses import margin_loss, seg_margin_loss
import matplotlib.pyplot as plt

def dumpclean(obj):
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print ('%s : %s' % (k, v))
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
    else:
        pass


# returns a compiled model
# identical to the previous one
model = load_model(
    args.model_file,
    custom_objects={
        'CoupledConvCapsule': CoupledConvCapsule,
        'CapsMaxPool': CapsMaxPool,
        'CapsuleNorm': CapsuleNorm,
        'CapsuleLayer': CapsuleLayer,
        'Length': Length,
        '_margin_loss': margin_loss(
            # these were used during training
            pos_margin=0.7, neg_margin=0.3
        ),
        '_seg_margin_loss': seg_margin_loss(),
    }
)
if args.dataset == 'cifar100':
    (X_train, _), (X_test, _) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
else:
    (X_train, _), (X_test, _) = tf.keras.datasets.cifar10.load_data()

if args.type == 'train':
    # images to visualize
    items = [3,5,11000,44000,35000,26000,13,999,1100,1200,37900,2900]
    folder_path = 'vis/'
    image_prefix = 'train_image'
else:
    items = [6,1,5,3,8500,200,1793,299,433,500,283,201,288,999,5001]
    folder_path = 'vis_test/'
    image_prefix = 'test_image'

item_prefixes = {
    'primarycap_reshape/Reshape:0': 'model9_primarycap_layer_item',
    'conv/Relu:0': 'model9_conv_layer_item'
}

import pathlib
path = pathlib.Path(folder_path)
path.mkdir(mode=0o766, parents=True, exist_ok=True)

for item_num in items:
    a = get_activations(model, X_train[item_num-1:item_num])  # with just one sample.
    plt.title('Train image #' + str(item_num))
    plt.imshow(X_train[item_num])
    plt.savefig(folder_path + image_prefix + str(item_num) + '.png', format='png')
    retrieved_activations = {}
    for k, v in a.items():
        if k == 'primarycap_reshape/Reshape:0':
            retrieved_activations[k] = v.reshape(1, 8, 8, 32 * 12)
            retrieved_activations[k] = retrieved_activations[k][:,:,:,:48]
        elif k == 'conv/Relu:0':
            retrieved_activations[k] = v[:,:,:,:48]

    dumpclean(a)

    display_activations(retrieved_activations, item_num, folder_path, item_prefixes)
