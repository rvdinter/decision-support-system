import numpy as np
from keras.constraints import unit_norm
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D

from metrics import WorkSavedOverSamplingAtRecall


def multichannel_cnn():
    """
    Build the Multichannel CNN model
    """
    feature_length = 600
    input_layer = Input(shape=(feature_length,))

    # Load the embedding weights and set the embedding layer
    weights = np.load(open('weights/100d.npy', 'rb'))
    embedding_layer = Embedding(input_dim=weights.shape[0],
                                output_dim=weights.shape[1],
                                mask_zero=False,
                                weights=[weights], trainable=False)

    embedding = embedding_layer(input_layer)
    embedding = Dropout(0.6)(embedding)

    # channel 1
    channel_1 = Conv1D(filters=1024, kernel_size=2, padding='valid', activation='relu')(embedding)
    channel_1 = GlobalMaxPooling1D()(channel_1)

    # channel 2
    channel_2 = Conv1D(filters=1024, kernel_size=4, padding='valid', activation='relu')(embedding)
    channel_2 = GlobalMaxPooling1D()(channel_2)

    # Fully connected network
    fully_connected = Concatenate()([channel_1, channel_2])
    fully_connected = Dropout(0.4)(fully_connected)
    fully_connected = Dense(128, activation='relu', kernel_constraint=unit_norm(), bias_constraint=unit_norm())(
        fully_connected)
    fully_connected = Dropout(0.4)(fully_connected)
    output = Dense(1, activation='sigmoid', kernel_constraint=unit_norm(), bias_constraint=unit_norm())(
        fully_connected)

    model = Model(inputs=(input_layer), outputs=output)

    # Model settings
    metrics = [WorkSavedOverSamplingAtRecall(recall=1, name='wss'),
               WorkSavedOverSamplingAtRecall(recall=0.95, name='wss_95')]
    opt = optimizers.Adam(1e-4)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=metrics)
    return model
