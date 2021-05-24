import numpy as np
import tensorflow as tf
import PIL
import time
import matplotlib.pyplot as plt

CONTENT_WEIGHT = 500.
STYLE_WEIGHT = 1e-2
VARIATION_WEIGHT = 30

# Preprocess an image
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Displays a singular image. Need to call plt.show() after
def display_img(image, title = None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

# Displays both the content and style image
def vis_data(content_path, style_path):
    plt.subplot(1, 2, 1)
    display_img(load_img(content_path), "Content Image")
    plt.subplot(1, 2, 2)
    display_img(load_img(style_path), "Style Image")
    plt.show()

# Used for computing the style loss
def gram_matrix(input_tensor):
    # output[b, c, d] = summation(input[b, i, j, c] * input[b, i, j, d]
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Combined loss of style and content
def content_style_loss(outputs, content_targets, style_targets, num_style_layers, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= STYLE_WEIGHT / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= CONTENT_WEIGHT / num_content_layers
    loss = style_loss + content_loss
    return loss

# Creates a vgg model that returns a list of intermediate output values.
def vgg_layers(layer_names):
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# Network that returns the style loss and content loss as a dictionary when called
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Expects float input in [0,1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

# Style transfer based on the gatys paper.
def gatys_style_transfer(content, style_path, epochs = 10, steps_per_epoch = 100):
    content_image = load_img(content)
    style_image = load_img(style_path)

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Create the loss network
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    # Generated image
    image = tf.Variable(content_image)
    images = [tf.Variable(content_image)]

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # Ensure image is between 0-1
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    # tf.function speeds up computation. This is the train step in our GD later
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = content_style_loss(outputs, content_targets, style_targets, num_style_layers, num_content_layers)
            loss += VARIATION_WEIGHT * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='')
        print("Train step: {}".format(step))
        #if display_every_epoch and n % display_every_epoch == 0:
        #display_img(image, "Train step: {}".format(step))
        #plt.show()
        images.append(tf.Variable(image))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    return image, images

