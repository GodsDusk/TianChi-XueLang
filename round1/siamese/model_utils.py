from keras.models import Model
from keras.layers import GlobalMaxPooling2D, Dropout, Dense, Lambda, Input
from keras import backend as K
from keras.applications.vgg19 import VGG19
# from keras.applications.xception import Xception
# from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam 

from config import *

def get_base_model():
    latent_dim = 50
    base_model = VGG19(weights='imagenet',include_top=False) # use weights='imagenet' locally
    # for layer in base_model.layers[:-2]:
    #     layer.trainable = False
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    dense_1 = Dense(latent_dim)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)#shape=(?, latent_dim)
    base_model = Model(base_model.input, normalized, name="base_model")
    return base_model 

def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)

def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))
    #shape=(?, 1)
    return loss

def build_model():
    base_model = get_base_model()

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)

    negative_example_out = base_model(negative_example)

    positive_example_2_out = base_model(positive_example_2)

    loss = Lambda(lambda x: bpr_triplet_loss(x))([positive_example_1_out,
                                                 negative_example_out,
                                                 positive_example_2_out])

    
    model = Model(
        inputs=[positive_example_1, negative_example, positive_example_2],
        outputs=loss)
    # model.load_weights('vgg19.hdf5')
    # model.compile(loss=identity_loss, optimizer='Adam')


    return model


def build_inference_model(weight_path=file_path):
    base_model = get_base_model()

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = Lambda(lambda x: bpr_triplet_loss(x))([positive_example_1_out,
                                                 negative_example_out,
                                                 positive_example_2_out])

    model = Model(
        inputs=[positive_example_1, negative_example, positive_example_2],
        outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))
    # model.compile(loss=identity_loss, optimizer='Adam')

    model.load_weights(weight_path)

    #Tensor("input_1:0", shape=(?, ?, ?, 3), dtype=float32) Tensor("lambda_1/l2_normalize:0", shape=(?, 50), dtype=float32)
    inference_model = Model(base_model.get_input_at(0), outputs=base_model.get_output_at(0))
    
    return inference_model




if __name__ == '__main__':

    # model = build_model()



    inference_model = build_inference_model(weight_path=file_path)