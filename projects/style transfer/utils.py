import numpy as np
from keras import backend as K
from keras.utils import Sequence
import cv2
import random

'''
Loading and preprocess utils
'''
def vgg_preprocess(x):
    # BGR Mean values
    vgg_mean = np.array([103.939, 116.779, 123.68], dtype='float32').reshape((1,1,3))
    return x - vgg_mean

def load_image(img, img_size):
    I = cv2.imread(img)
    I = cv2.resize(I, img_size)
    return I

def get_output(model, block, conv_n):
    return model.get_layer('block{}_conv{}'.format(block, conv_n)).output

# We need to write our own generator because the model's outputs are the losses
# Hence, the "targets" (or ground truths) for our datasets are just a bunch of 0s (1 for each loss output)
class ImageLoader(Sequence):
    def __init__(self, files, img_size = (256, 256), batch_size = 16): #, flip=False):
        self.files = files
        self.batch_size = batch_size
        self.img_size = img_size
        #self.flip = flip

    #gets the number of batches this generator returns
    def __len__(self):
        l,rem = divmod(len(self.files), self.batch_size)
        return (l + (1 if rem > 0 else 0))

    #shuffles data on epoch end
    def on_epoch_end(self):
        random.shuffle(self.files)

    #gets a batch with index = i
    def __getitem__(self, i):
        images = self.files[i*self.batch_size : (i+1)*self.batch_size]
        x = [load_image(f, self.img_size) for f in images]
        x = np.stack(x, axis=0)

        # Add targets: 3 loss functions so we add 3 0s as our targets
        targets = np.zeros((len(x),1))
        return x.astype('float32'), [targets]*3

'''
Loss computation utils
'''
def rmse(y, y_pred):
    return K.expand_dims(K.sqrt(K.mean(K.square(y - y_pred), [1,2,3])), 0)

# Used for computing the style loss
def gram_matrix(x):
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    bs, c, h, w = K.int_shape(x)
    features = K.reshape(x, (-1, c, h*w))
    features_T = K.permute_dimensions(features, (0, 2, 1))
    gram = K.batch_dot(features, features_T)
    return gram



