import style_transfer as st
import fast_style_transfer as fst
import matplotlib.pyplot as plt
from net import ReflectionPadding2D
from keras.models import load_model
import cv2
import os
import math

def main():
    # 0 for my own trained weights and 1 for the downloaded weights
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    style_path = "armchair.jpg"
    weights_path = "weights/armchair.h5"
    shape = (256, 256)

    # Uncomment for torture
    #fst.train(dir_path, style_path)

    #Uncomment to run gatys slow style transfer
    # img, images = st.gatys_style_transfer(dir_path + "/img/dog.jpg", dir_path +"/img/" +style_path, 3, 100)
    # st.display_img(img, "Final Image")
    # plt.show()
    #
    # for i in range(len(images)):
    #     st.plt.xticks([])
    #     st.plt.yticks([])
    #     st.display_img(images[i], "Steps: {}".format(i * 100))
    #     plt.show()
    # #st.plt.show()
    # st.tensor_to_image(img).save("generated_img.png")
    #
    # # Uncomment to run downloaded weights
    # style_image = cv2.imread(dir_path + "/img/" + style_path)
    # plt.imshow(style_image[:, :, ::-1])  # bgr->rgb
    # plt.draw()
    #
    # # Load the weights for the image transformation network
    generator = load_model(dir_path + weights_path,
                           custom_objects={'ReflectionPadding2D': ReflectionPadding2D})

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while (ret):
        frame = cv2.flip(frame, 1)
        img = cv2.resize(frame, shape)[None]
        pred = generator.predict_on_batch(img).astype('uint8')[0]

        frame = cv2.resize(frame, (640, 480))
        pred = cv2.resize(pred, (640, 480))

        cv2.imshow('Original', frame)
        cv2.imshow('Styled', pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    # plt.show()
main()