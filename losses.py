from SegCaps.custom_losses import margin_loss as segcaps_margin_loss
import keras.backend as K

def margin_loss(downweight=0.5, pos_margin=0.9, neg_margin=0.1):
  def _margin_loss(y_true, y_pred):
    """
    Taken from https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py#L80
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule (n_classes)]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., pos_margin - y_pred)) + \
        downweight * (1 - y_true) * K.square(K.maximum(0., y_pred - neg_margin))

    return K.sum(L, 1)
  return _margin_loss

def seg_margin_loss():
  _segcaps_margin_loss = segcaps_margin_loss()
  def _seg_margin_loss(y_true, y_pred):
    """Margin loss using one defined in SegCaps implementation
    
    Wrap around SegCaps implementation with default parameters and return a single scalar
    
    :param y_true: True y labels
    :type y_true: Tensor, shape: [None, n_classes]
    :param y_pred: Prediction y labels
    :type y_pred: Tensor, shape [None, num_capsule (n_classes)]
    """
    loss = _segcaps_margin_loss(y_true, y_pred)
    return K.sum(loss, 1)
  return _seg_margin_loss