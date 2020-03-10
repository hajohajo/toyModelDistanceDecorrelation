import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

scale = 1.07862
alpha = 2.90427
def serlu(x):
    return scale * tf.where(x >= 0.0, x, alpha * x * tf.exp(x))

class InputSanitizerLayer(tf.keras.layers.Layer):
    def __init__(self, minValues, maxValues, **kwargs):
        self.minValues = minValues
        self.maxValues = maxValues

        self.tensorMins = tf.convert_to_tensor(np.reshape(self.minValues, (1, self.minValues.shape[-1])), dtype='float32')
        self.tensorMaxs = tf.convert_to_tensor(np.reshape(self.maxValues, (1, self.maxValues.shape[-1])), dtype='float32')
        super(InputSanitizerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InputSanitizerLayer, self).build(input_shape)

    def call(self, input):
        values = tf.math.multiply(tf.math.divide((tf.math.maximum(tf.math.minimum(input, self.tensorMaxs), self.tensorMins) - self.tensorMins), (self.tensorMaxs - self.tensorMins)), 2) - 1.0
        return values

    def get_config(self):
        return {'minValues': self.minValues, 'maxValues': self.maxValues}

from discoTf import distance_corr
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils

#DiscoLoss expects that the "target" is in the shape [batch_size, 3], where the columns
# 0: The actual target i.e. signal or background
# 1: The variable of interest to decorrelate from i.e. TransverseMass
# 2: Weights required for the distance correlation loss

# The weights proposed in https://arxiv.org/abs/2001.05310 are such that the signal events
# are given zero weight (as we don't explicitly care if they are decorrelated), and the
# background samples are given weights that approximately flatten their distribution w.r.t
# the variable of interest, and sum up to the total number of events (including signal).
# Function discoWeights provides these if necessary
class DiscoLoss(Loss):
    def __init__(self, factor=1.0):
        self.factor = factor
        self.name = "DiscoLoss"
        self.reduction = losses_utils.ReductionV2.AUTO

    def call(self, y_true, y_pred):
        # Split given labels to the target and the mT value needed for decorrelation
        y_pred = tf.convert_to_tensor(y_pred)
        sample_weights = tf.cast(tf.reshape(y_true[:, 2], (-1, 1)), y_pred.dtype)
        mt = tf.cast(tf.reshape(y_true[:, 1], (-1, 1)), y_pred.dtype)
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)

        dcPred = tf.reshape(y_pred, [tf.size(y_pred)])
        dcMt = tf.reshape(mt, [tf.size(mt)])
        weights = tf.cast(tf.reshape(sample_weights, [tf.size(sample_weights)]), y_pred.dtype)

        # The loss
        custom_loss = tf.add(tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0),
                             self.factor * distance_corr(dcMt, dcPred, normedweight=weights, power=1))
        return custom_loss


#AUC that can be used as a metric when using the discoLoss with 3D target
def get_custom_auc():
    auc = tf.metrics.AUC()
    # @tf.function
    def custom_auc(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)
        auc.update_state(y_true, y_pred)
        return auc.result()

    custom_auc.__name__ = "custom_auc"
    return custom_auc

def get_custom_crossentropy():
    def custom_crossentropy(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)
        return tf.losses.binary_crossentropy(y_true, y_pred)

    return custom_crossentropy

def get_disco():

    @tf.function(experimental_relax_shapes=True)
    def DisCo_metric(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        mt = tf.cast(tf.reshape(y_true[:, 1], (-1, 1)), y_pred.dtype)
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)

        #Type casting and reshaping
        numEntries = tf.cast(tf.size(y_true), dtype=tf.float32)
        numSignalEntries = tf.cast(tf.reduce_sum(y_true), dtype=tf.float32)
        bitMaskForBkg = tf.cast(tf.logical_not(tf.cast(y_true, dtype=tf.bool)), dtype=tf.float32)

        #Calculate the weights only for background, while setting signal weights to zero (we're only
        #interested in decorrelating the background from mT, signal doesn't matter)
        #Note that distance_corr function expects the weights to be normalized to the number of events
        weightFactor = tf.divide(numEntries, tf.subtract(numEntries, numSignalEntries))
        weights = tf.multiply(weightFactor, bitMaskForBkg)

        dcPred = tf.reshape(y_pred, [tf.size(y_pred)])
        dcMt = tf.reshape(mt, [tf.size(mt)])
        weights = tf.cast(tf.reshape(weights, [tf.size(weights)]), y_pred.dtype)

        # return distance_corr(dcMt, dcPred, normedweight=weights, power=1)
        return distance_corr(dcMt, dcPred, normedweight=tf.ones(tf.shape(dcMt)), power=1)

    return DisCo_metric

def createClassifier(nInputs, minValues, maxValues):
    _neurons = 16
    _activation = serlu
    _regularization = tf.keras.regularizers.l2(1e-2)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        input = keras.layers.Input(nInputs)
        x = InputSanitizerLayer(minValues, maxValues)(input)
        x = keras.layers.Dense(_neurons, activation=_activation, kernel_regularizer=_regularization)(x)
        x = keras.layers.Dense(_neurons, activation=_activation, kernel_regularizer=_regularization)(x)
        x = keras.layers.Dense(_neurons, activation=_activation, kernel_regularizer=_regularization)(x)
        x = keras.layers.Dense(_neurons, activation=_activation, kernel_regularizer=_regularization)(x)
        x = keras.layers.Dense(_neurons, activation=_activation, kernel_regularizer=_regularization)(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(input, output)

        model.compile(optimizer=keras.optimizers.Adam(lr=1e-3, amsgrad=True),
                      metrics=[get_custom_auc(), get_disco(), get_custom_crossentropy()],
                      loss=DiscoLoss(250.0))

        return model
