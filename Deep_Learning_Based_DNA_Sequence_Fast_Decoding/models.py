import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, \
    BatchNormalization, Bidirectional, LSTM, Dropout, Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
    AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling1D, MultiHeadAttention,\
    LayerNormalization, Embedding, LeakyReLU, Conv1DTranspose


def cnn_model(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Conv1D(32, 17, padding='same', activation='relu'),
        Conv1D(64, 11, padding='same', activation='relu'),
        Conv1D(128, 5, padding='same', activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

## Step 1: Implement your own model below

def pcc(layer_in, num_filter, size_kernel, activation='relu', padding='same'):
    x = MaxPooling1D(pool_size=2, padding=padding)(layer_in)
    x = BatchNormalization()(Conv1D(num_filter, size_kernel,
                                    activation=activation, padding=padding)(x))
    x = BatchNormalization()(Conv1D(num_filter, size_kernel,
                                    activation=activation, padding=padding)(x))
    return x


def ucc(layer_in1, layer_in2, num_filter, size_kernel, activation='relu', padding='same'):
    x = concatenate([Conv1DTranspose(num_filter, 2,
                    strides=2, padding=padding, activation=activation)(layer_in1), layer_in2], axis=2)
    x = BatchNormalization()(Conv1D(num_filter, size_kernel,
                                    activation=activation, padding=padding)(x))
    x = BatchNormalization()(Conv1D(num_filter, size_kernel,
                                    activation=activation, padding=padding)(x))
    return x


def leopard_unet_model(max_len, vocab_size):
    inputs = Input((max_len, vocab_size))
#    print(inputs.shape)

    num_blocks = 5 
    initial_filter = 15
    scale_filter = 1.5
    size_kernel = 7
    activation = 'relu'
    padding = 'same'

    layer_down = []
    layer_up = []

    conv0 = BatchNormalization()(Conv1D(initial_filter, size_kernel,
                                        activation=activation, padding=padding)(inputs))
    conv0 = BatchNormalization()(Conv1D(initial_filter, size_kernel,
                                        activation=activation, padding=padding)(conv0))

    layer_down.append(conv0)
    num = initial_filter

    for i in range(num_blocks):
        num = int(num * scale_filter)
        the_layer = pcc(layer_down[i], num, size_kernel,
                        activation=activation, padding=padding)
        layer_down.append(the_layer)

    layer_up.append(the_layer)
    for i in range(num_blocks):
        num = int(num / scale_filter)
        the_layer = ucc(layer_up[i], layer_down[-(i+2)], num,
                        size_kernel, activation=activation, padding=padding)
        layer_up.append(the_layer)

    convn = Conv1D(1, 1, activation='sigmoid', padding=padding)(layer_up[-1])

    model = Model(inputs=[inputs], outputs=[convn])

    return model

## Step 2: Add your model name and model initialisation in the model dictionary below

def return_model(model_name, max_len, vocab_size):
    model_dic={'cnn': cnn_model(max_len, vocab_size), 'leopard_unet': leopard_unet_model(max_len, vocab_size)}
    return model_dic[model_name]



