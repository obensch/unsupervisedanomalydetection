import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.initializers import Constant
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, PReLU, BatchNormalization, Activation
from keras.models import Model
from keras.utils.vis_utils import plot_model

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# patch size
patch_size = 64

# model settings
train_num = 1000000
val_num = 200000
epochs_num = 3
batch_size = 1000
c = 32
latentDim=256

patchString = "_" + str(patch_size)
patch_path = "data/patches_his" + patchString + "/"

model_name = str(patch_size) + "_LeakyReLU_1mT_200kV_L" + str(latentDim) + "_E" + str(epochs_num) + "_C" + str(c) + ".md"
model_path = "data/models/"

# build the model
def build_model():
    chanDim = -1

    inputs = Input(shape=(patch_size, patch_size, 1))

    x = Conv2D(c, (3, 3), strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

    x = Conv2D(2*c, (3, 3), strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

    # x = Conv2D(3*c, (3, 3), strides=2, padding="same")(x)
    # x = LeakyReLU(alpha=0.2)(x)
    # x = BatchNormalization(axis=chanDim)(x)

    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latentDim)(x)

    encoder = Model(inputs, latent, name="encoder")

    latentInputs = Input(shape=(latentDim,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    # x = Conv2DTranspose(3*c, (3, 3), strides=2, padding="same")(x)
    # x = LeakyReLU(alpha=0.2)(x)
    # x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(2*c, (3, 3), strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(c, (3, 3), strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(1, (3, 3), padding="same")(x)
    outputs = Activation("sigmoid")(x)

    decoder = Model(latentInputs, outputs, name="decoder")
    

    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

    # plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
    # plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)
    # plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # encoder.summary()
    # decoder.summary()
    # autoencoder.summary()
    return autoencoder

# data generator to load patches
class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, start=0, end=1000000):
        self.start, self.end = start, end
        self.total = end - start
        self.indexes = np.arange(self.total)
        self.batch_size = batch_size
        self.shuffle = False
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.total / self.batch_size)
    
    def __getitem__(self, idx):
        idxs = [i for i in range(idx*self.batch_size,(idx+1)*self.batch_size)]
        batch_X = np.zeros((self.batch_size, patch_size, patch_size))
        batch_Y = np.zeros((self.batch_size, patch_size, patch_size))
        for i in range(self.batch_size):
            x = np.load(patch_path + str(idxs[i])+".npy")
            batch_X[i] = x.astype('float32') / 255
            batch_Y[i] = x.astype('float32') / 255
        batch_X.reshape((self.batch_size, patch_size, patch_size, 1))
        batch_Y.reshape((self.batch_size, patch_size, patch_size, 1))
        return batch_X, batch_Y

    def on_epoch_end(self):
        self.indexes = np.arange(self.total)

TrainGen = MyDataGenerator(start=0, end=train_num)
ValGen = MyDataGenerator(start=train_num, end=(train_num+val_num))

autoencoder = build_model()
autoencoder.compile(optimizer='adam', loss='mse', metrics=["acc"])

history = autoencoder.fit(TrainGen, validation_data=ValGen, epochs=epochs_num)
autoencoder.save(model_path + model_name)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
