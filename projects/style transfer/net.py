from utils import *
from keras.models import Model
from keras.layers import *
import tensorflow as tf
#from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model

# From https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
# Used to load downloaded weights as well as own model
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    # So that tensorflow can re-instantiate this layer after saving
    def get_config(self):
        config = {
            'padding': self.padding,
            'input_spec': self.input_spec,
        }
        return config


# Adds a convolution block
def conv_block(x, num_filters, kernel_size=3, stride=1, activation=True):
    x = Conv2D(num_filters, kernel_size, strides=stride, use_bias = False)(x)
    if activation:
        x = Activation('relu')(x)
    x = BatchNormalization()(x)
    return x


# Adds a residual block
def res_block(x, num_filters):
    res = conv_block(x, num_filters)
    res = conv_block(res, num_filters)
    cropped_x = Cropping2D(cropping=((2, 2), (2, 2)))(x)
    # input tensors must be same size
    return add([cropped_x, res])


'''
This is our image transformation net. It sits on top of our loss network during training and we train this neural net 
to transform an input image such that it fits the style that it is trained on. 
Each style has to be trained separately and will have different weights.
'''
def image_transform_net(shape):
    inp = Input(shape)
    d0 = Lambda(lambda x: x / 127.5 - 1)(inp)
    d0 = ReflectionPadding2D(padding=(54, 54))(d0)

    # As per the paper, we have 3 conv blocks followed by 5 residual blocks and lastly 3 deconvolution blocks
    # Convolution blocks
    x = conv_block(d0, 32, 9)
    x = conv_block(x, 64, stride=2)
    x = conv_block(x, 128, stride=2)

    # Residual blocks
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)

    # Deconvolution Blocks/reverse MaxPooling
    x = UpSampling2D()(x)
    x = conv_block(x, 64)
    x = UpSampling2D()(x)
    x = conv_block(x, 32)
    x = conv_block(x, 3, 11, activation=False)
    x = Activation('tanh')(x)
    output_img = Lambda(lambda x: 127.5 * (x + 1))(x)
    gen = Model(inp, output_img, name='Style_Generator')
    return gen


'''
This is our training model. It trains the Image Transform Net defined above by reducing the loss computed by VGG16.
The model consists of 2 parts: The image transformation net (the net we want to train), and the loss net (not trained).
The image transformation net outputs an image based on a specific style and passes it to the loss network.
The loss network then computes the style, content and variation loss as outputs. 
Finally, the model will compare the true losses comparing the above 3 outputs with our targets which is just [0,0,0].
'''
def training_model(img_rows, img_cols, channels, style_image_path, plot_models=False, dir_path = None):
    image_shape = (img_rows, img_cols, channels)
    style_image = cv2.imread(style_image_path)
    style_image = cv2.resize(style_image, image_shape[:2])
    tf_style = tf.convert_to_tensor(style_image[None], dtype='float32')  # tensor to evaluate gram matrix

    input = Input((img_cols, img_rows, channels), name='InputImage')
    vgg = VGG16(include_top=False, weights='imagenet',
                input_tensor=Lambda(vgg_preprocess)(input))
    for l in vgg.layers: l.trainable = False

    content_extractor = Model(input, get_output(vgg, 5, 2), name='Content-VGG')
    style_extractor = Model(input, [get_output(vgg, o, 1) for o in [1, 2, 3, 4, 5]], name='Style-VGG')
    input_content = content_extractor(input)            # Get output of the original image
    style_image_outputs = style_extractor(tf_style)     # get the style layer outputs for the style image

    # This is the part we train. The other parts are not trained. This model generates stylized images.
    transform_net = image_transform_net(shape=(img_cols, img_rows, channels))
    transformed_image = transform_net(input)                    # Input -> Image Transform Net
    transformed_content = content_extractor(transformed_image)  # Image Transform Net -> VGG Content
    transformed_style = style_extractor(transformed_image)      # Image Transform Net -> VGG Style

    # Compute gram matrix for style image
    style_matrix = []
    for i in range(len(style_image_outputs)):
        style_matrix.append(gram_matrix(style_image_outputs[i]))

    # Only compute style matrix for style image once
    def style_loss(x):
        s_loss = 0
        n = len(x)
        for i in range(n):
            _, w, h, channels = K.int_shape(x[i])
            size = w * h
            C = gram_matrix(x[i])  # calculate the gram matrix of every VGG's layer output
            loss = K.sum(K.square(style_matrix[i] - C), axis=[1, 2]) / (4.0 * (channels ** 2) * (size ** 2))
            s_loss += loss / n
        return K.expand_dims(s_loss, 0)

    def content_fn(x):
        content_loss = 0
        n = len(x) // 2
        for i in range(n): content_loss += rmse(x[i], x[i + n]) / n
        return content_loss

    def total_variation_loss(x):
        a = K.square(
            x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
        b = K.square(
            x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
        return K.expand_dims(K.sum(K.pow(a + b, 1.25), axis=[1, 2, 3]), 0)

    # Create Loss Output Layers
    outputs = [
        Lambda(content_fn, name='Content-Loss')([input_content, transformed_content]),   # VGG Content -> Content Loss
        Lambda(style_loss, name='Style-Loss')(transformed_style),                        # VGG Style -> Style Loss
        Lambda(total_variation_loss, name='Variation-Loss')(transformed_image)  # Image Transform Net -> Variation Loss
    ]

    model = Model(input, outputs, name='Style-Transfer')

    if plot_models:
        plot_model(transform_net, dir_path + "image_gen_model.png", show_shapes=True)
        plot_model(model, dir_path + "my_final_model.png", show_shapes=True)

    return model, transform_net
