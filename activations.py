import tensorflow as tf
from collections import Callable

def non_saturating_squash(input_tensor):
	norm = tf.norm(input_tensor, axis=-1, keepdims=True)
	return input_tensor * (norm / (1 + norm))

def callable(obj):
	return isinstance(obj, Callable)

# this is done because of a mistake in the initial design
# of the program, but no time to change
def get_capsule_activation(activation):
	if activation == 'squash':
		return layers.capsule.squash
	elif activation == '_squash':
		return SegCaps.capsule_layers._squash
	elif activation == 'non_saturating_squash':
		return non_saturating_squash
	elif callable(activation):
		# assume right
		return activation
	else:
		raise Exception(activation + ' is not an activation for capsules')

import layers.capsule
import SegCaps.capsule_layers