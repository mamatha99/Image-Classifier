import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import json
import argparse
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import logging
logger=tf.get_logger()
logger.setLevel(logging.ERROR)


parser=argparse.ArgumentParser(description='description for arg parser')
parser.add_argument("image_path",action="store")
parser.add_argument("saved_model",action="store")
parser.add_argument("--top_k",action="store",dest="top_k",default=5,required=False)
parser.add_argument("--category_names",action="store",dest="category_names")
results=parser.parse_args()
image_path=results.image_path
saved_model=results.saved_model
category_filename=results.category_names

if results.top_k==None:
    top_k=5
else:
    top_k=results.top_k
    
def process_image(image):
    image_size=224
    image=tf.cast(image, tf.float32)
    image = tf.image.resize(image,(image_size,image_size))
    image /= 255
    image=image.numpy()
    return image

def predict_image(image_path, model, top_k=5):
    im = Image.open(image_path)
    testing_image=np.asarray(im)
    processed_image=process_image(testing_image)
    final_image=np.expand_dims(processed_image,axis=0)
    prob_preds=model.predict(final_image)
    probs= -np.partition(-prob_preds[0],top_k)[:top_k]
    classes= np.argpartition(-prob_preds[0],top_k)[:top_k]
    return probs,classes


model=tf.keras.models.load_model(saved_model
                                 ,custom_objects={'KerasLayer':hub.KerasLayer})
image = np.asarray(Image.open(image_path)).squeeze()[0]
probs,classes=predict_image(image_path, model, top_k)  

if category_filename==None:
    with open('label_map.json', 'r') as f:
         class_names = json.load(f)
    keys=[str(x+1) for x in list(classes)]
    classes=[class_names.get(key) for key in keys]
else:
    with open(category_filename, 'r') as f:
         class_names = json.load(f)
    keys=[str(x+1) for x in list(classes)]
    classes=[class_names.get(key) for key in keys]

print('top classes'.format(top_k))
for i in np.arange(top_k):
    print("classes:{}".format(classes[i]))
    print("probability:{:.4%}".format(probs[i]))
    print("\n")
    
