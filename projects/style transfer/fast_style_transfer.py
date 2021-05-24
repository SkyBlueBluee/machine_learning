from net import *
from glob import glob

# Global Params
batch_size = 4
epochs = 2

# Hyper Parameters
CONTENT_WEIGHT = 10000.
STYLE_WEIGHT = 0.001
TOTAL_VARIATION_WEIGHT = 1e-6
WEIGHTS = [CONTENT_WEIGHT, STYLE_WEIGHT, TOTAL_VARIATION_WEIGHT]


'''
Trains the image transformation network to output a new image based on a certain style.
'''
def train(dir_path, style_name):
    img_rows = 256
    img_cols = 256
    channels = 3
    img_shape = (img_rows, img_cols, channels)

    train_path = dir_path + "train/*"
    style_path = dir_path + "img/" + style_name

    # Uncomment to show the style image before training
    # style_image = cv2.imread(style_path)
    # style_image = cv2.resize(style_image, image_shape[:2])
    # plt.imshow(style_image[:, :, ::-1])  # bgr->rgb
    # plt.show()

    model, transform_net = training_model(img_rows, img_cols, channels, style_path, plot_models=True, dir_path=dir_path)
    data_gen = ImageLoader(glob(train_path), img_shape[:2], batch_size)
    print("Images found: {}".format(len(data_gen) * batch_size))
    print("Number of Batches: {}".format(len(data_gen)))

    model.compile(optimizer='adam', loss=['mse', 'mse', 'mse'], loss_weights=WEIGHTS)
    print(model.summary())
    model.fit(data_gen, steps_per_epoch=len(data_gen), epochs=epochs)

    transform_net.save(dir_path + "weights/" + style_name[:-4] + '.h5')
    print('Saved Model {} Weights.'.format(style_name[:-4]))


'''
This was an attempt at writing my custom training loop to apply the concepts discussed in the following paper:
https://medium.com/element-ai-research-lab/stabilizing-neural-style-transfer-for-video-62675e203e42
The key concept is that when we add noise to the raw input and pass that into the image transformation net.
Then, we compare the predicted noisy output and the normal output by the image transformation net and compute a
new loss value based on the differences. This should stabilise the "popping" that most NSTs suffer from.
'''
def custom_train(dir_path, style_name):
    image_rows = 672 // 2  # 672, 376
    image_cols = 376 // 2
    channels = 3
    image_shape = (image_rows, image_cols, channels)

    train_path = dir_path + "train/*"
    style_path = dir_path + "img/" + style_name

    style_image = cv2.imread(style_path)
    style_image = cv2.resize(style_image, image_shape[:2])
    tf_style = tf.convert_to_tensor(style_image[None], dtype='float32')  # tensor to evaluate gram matrix

    input = Input(image_shape, name='Input-Image')
    vgg = VGG16(include_top=False, weights='imagenet',
                input_tensor=Lambda(vgg_preprocess)(input))
    for l in vgg.layers: l.trainable = False

    content_extractor = Model(input, get_output(vgg, 5, 2), name='Content-VGG')
    style_extractor = Model(input, [get_output(vgg, o, 1) for o in [1, 2, 3, 4, 5]], name='Style-VGG')
    input_content = content_extractor(input)  # content of original image
    style_image_outputs = style_extractor(tf_style)  # get the style layer outputs for the style image

    # compute gram matrix for style image
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

    # This is the part we train. The other parts are not trained
    transform_net = image_transform_net(shape=image_shape)  # load generator model - generating styled images
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    data_gen = ImageLoader(glob(train_path), image_shape[:2], batch_size)

    # tf.function speeds up computation. This is the train step in our GD later
    @tf.function()
    def train_step(input):
        with tf.GradientTape() as tape:
            output = transform_net(input)
            content_input = content_extractor(input)
            content_pred = content_extractor(output)
            style = style_extractor(output)

            c_loss = content_fn([content_input, content_pred])
            s_loss = style_loss(style)
            v_loss = total_variation_loss(output)
            loss = c_loss * CONTENT_WEIGHT + s_loss * STYLE_WEIGHT + v_loss * TOTAL_VARIATION_WEIGHT
        grads = tape.gradient(loss, transform_net.trainable_weights)
        opt.apply_gradients(zip(grads, transform_net.trainable_weights))

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        #start_time = time.time()
        steps_per_epoch = len(data_gen) // batch_size
        # Iterate over the batches of the dataset.
        for step in range(steps_per_epoch):
            x, y = data_gen.__getitem__(step)
            loss_value = train_step(x)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * 64))
        data_gen.on_epoch_end()

    transform_net.save(dir_path + "weights/" + style_name[:-4] + '.h5')
    print('Saved Model {} Weights.'.format(style_name[:-4]))