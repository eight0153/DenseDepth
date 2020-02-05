import os

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import plac
from PIL import Image
from keras.models import load_model

from layers import BilinearUpSampling2D
from utils import predict

plt.set_cmap("gray")


def load_image(img_path):
    image = Image.open(img_path)
    image = image.resize((640, 480))
    image = np.array(image)

    imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    image = (image - imagenet_stats['mean']) / imagenet_stats['std']
    image = image / 255
    image = np.clip(image, 0, 1)

    return image



@plac.annotations(
    image_path=plac.Annotation('The path to an RGB image or a directory containing RGB images.', type=str,
                               kind='option', abbrev='i'),
    model_path=plac.Annotation('The path to the pre-trained model weights.', type=str, kind='option', abbrev='m'),
    output_path=plac.Annotation('The path to save the model output to.', type=str, kind='option', abbrev='o'),
)
def main(image_path, model_path='nyu.h5', output_path=None):
    print("Loading model...")

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    print("Creating depth maps...")
    rgb_path = os.path.abspath(image_path)

    if os.path.isdir(rgb_path):
        for file in os.listdir(rgb_path):
            test(model, os.path.join(rgb_path, file), output_path)
    else:
        test(model, rgb_path, output_path)

    print("Done.")


def test(model, rgb_path, output_path):
    path, file = os.path.split(rgb_path)
    file = f"{file.split('.')[0]}.png"
    depth_path = os.path.join(output_path, file) if output_path else os.path.join(path, f"out_{file}")

    print(f"{rgb_path} -> {depth_path}")

    image = load_image(rgb_path)
    out = predict(model, image)

    matplotlib.image.imsave(depth_path, out[0].squeeze())


if __name__ == '__main__':
    plac.call(main)
