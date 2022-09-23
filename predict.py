import argparse
import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(
            description = 'Given an image and a pretrained model, predict top K class labels of the image')

# Load image from the path
def load_image(image_path: str):
    img = Image.open(image_path)
    img = np.asarray(img)
    return img

# Image processing
def process_image(image: np.array, 
                  image_size: int = 224) -> np.array:
    '''Function to process a given image
       before feeding into the model'''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

#predict function
def predict(img: np.array, 
            model: tf.keras.models, 
            top_k: int = 5) -> Tuple:
    '''Return the `top_k` most probable classes of the given
       image using a pretraining model'''
    img = np.expand_dims(img, axis = 0)
    preds   = model.predict(img)[0]
    top_values_idx = np.argpartition(preds, -top_k)[-top_k:]
    top_values_idx = top_values_idx[np.argsort(preds[top_values_idx])][::-1]
    probs   = preds[top_values_idx]
    # add one so the first class will be 1 instead of 0
    classes = top_values_idx + 1 
    classes = [str(c) for c in classes]
    return probs, classes

# Check that the inputs are correct
def check_image_path(image_path: str) -> str:
    if not os.path.isfile(f'./{image_path}'):
        raise argparse.ArgumentTypeError(
            "Please provide a valid path to the image.")
    return image_path

def check_model_path(model_path: str) -> str:
    if not os.path.isfile(f'./{model_path}'):
        raise argparse.ArgumentTypeError(
            "Please provide a valid `h5` file with a pretrained `tf.keras` model.")
    return model_path

def check_labels_path(category_names_json: str) -> str:
    if not os.path.isfile(f'./{category_names_json}'):
        raise argparse.ArgumentTypeError(
            "Please provide a valid `category_names_json` file.")
    return category_names_json

# Parser arguments
parser.add_argument('-i', '--image_path', 
                    required = True, type = check_image_path,
                    help = 'Path to the flower image to be predicted.')
parser.add_argument('-m', '--model_path', 
                    required = True, type = check_model_path,
                    help = 'Path to the pretrained model (.h5 format).')
parser.add_argument('-k', '--top_k_classes', type=int,
                    required = False, default = 5,
                    help = 'Return the top `k` most likely classes.')
parser.add_argument('-c', '--category_names_json', 
                    required = False, default = 'label_map.json',
                    help = 'A json file mapping the category names.')

# Capture the input arguments
args = vars(parser.parse_args())
IMAGE_PATH = args['image_path']
MODEL_PATH = args['model_path']
TOP_K      = args['top_k_classes']
LABEL_MAP  = args['category_names_json']


if __name__ == '__main__':
    # Load the image
    img = load_image(IMAGE_PATH)
    img_array = process_image(img)
    
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'KerasLayer': hub.KerasLayer}, 
                                       compile = False)
    
    #  Make the prediction
    probs, pred_classes = predict(img_array, model = model, top_k = TOP_K)
    
    # Load class names
    with open(LABEL_MAP, 'r') as f:
        all_class_names = json.load(f)

    #  Map the predicted labels and print the predictions
    pred_class_names = [all_class_names.get(key) 
                        for key in pred_classes]
    
    
    print(f'Model predictions for image {IMAGE_PATH}:')
    print('---'*20)
    for c, p in zip(pred_class_names, probs):
        print(c)
        print('- Probability:', round(p, 4))
        #print('-', '|' * int(p*50))
        
        
