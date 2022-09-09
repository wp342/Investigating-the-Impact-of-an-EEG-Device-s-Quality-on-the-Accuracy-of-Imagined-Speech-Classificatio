# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Lambda
import tensorflow as tf
import tensorflow.keras.backend as K


# relu used after every convolutional and fully connected layers
def make_embedding(inp1,inp2,inp3):
    inp = Input(shape=(inp1,inp2,inp3), dtype='float32')

    # First block
    c1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=5)(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (1, 3), activation='relu', padding='same')(m1)
    m2 = MaxPooling2D((1, 3), padding='same', strides=1)(c2)

    # Third block
    c3 = Conv2D(128, (1, 2), activation='relu', padding='same')(m2)
    m3 = MaxPooling2D((1, 2), strides=1, padding='same')(c3)

    f1 = Flatten()(m3)

    # fully connected layers
    d1 = Dense(1024, activation='relu')(f1)
    d2 = Dense(512)(d1)
    d3 = Dense(256)(d2)
    d4 = Dense(8)(d3)
    # d4 = Dense(1024, activation='relu')(d3)

    return Model(inputs=[inp], outputs=[d4], name='embedding')


# # Siamese L1 Distance class
# class L1Dist(Layer):
#
#     # Init method - inheritance
#     def __init__(self, **kwargs):
#         super().__init__()
#
#     # Magic happens here - similarity calculation
# def euclid_dis(input_embedding):
# return tf.norm(input_embedding[0] - input_embedding[1], ord='euclidean') #tf.math.abs(input1_embedding - input2_embedding)

class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, featsA, featsB, **kwargs):
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                           keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))
       #return tf.math.abs(input_embedding - validation_embedding)

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def make_siamese_model(inp1,inp2,inp3):
    # Anchor image input in the network
    eeg1 = Input(name='eeg_sample_1', shape=(inp1,inp2,inp3), dtype='float32')

    # Validation image in the network
    eeg2 = Input(name='eeg_sample_2', shape=(inp1,inp2,inp3), dtype='float32')

    # Combine siamese distance components
    embedding = make_embedding(inp1,inp2,inp3)
    # siamese_layer = Lambda
    # siamese_layer._name = 'distance'

    siamese_layer = L1Dist()
    embedding1 = embedding(eeg1)
    embedding2 = embedding(eeg2)
    distances = siamese_layer(embedding1, embedding2)
   # d1 = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[eeg1, eeg2], outputs=[distances, embedding1, embedding2], name='SiameseNetwork')


# make_embedding().summary()

#make_siamese_model().summary()
