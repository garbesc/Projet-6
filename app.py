import os, time
import pickle
import base64
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model


@st.cache_resource
def load_model(model_path):
    model = load_model(model_path)
    return model

#def load_image(img):
#    """ transform into array and preprocess image """
#    img = img.resize((299,299), Image.ANTIALIAS)
#    img_tensor = image.img_to_array(img)
#    img_tensor = np.expand_dims(img_tensor, axis=0)
#    img_tensor = preprocess_input(img_tensor)
#    return img_tensor

#def get_prediction(model, img, class_names):
#    """ Make prediction using model """
#    preds = model.predict(img)
#    pred_label = class_names[np.argmax(preds)]
#    return pred_label

def main():
    image = Image.open('logo.png')
    model_path = "/Users/claude/Desktop/Python/OpenClassrooms/Projet n°6 Classez des images/models/model_xception.h5"
    images_path = "/Users/claude/Desktop/Python/OpenClassrooms/Projet n°6 Classez des images/Images/"

    # load the Xception model
    model = load_model(model_path)
    
#    model = load_model(model_path)
    class_names  = []
    for path, dirs, files in os.walk(images_path):
        class_names.extend(dirs)
        
    st.image(image, width=100)
    st.title("Détection de la race d'un chien")

    uploaded_file = st.file_uploader("Veuillez choisir une image")  

    if uploaded_file is not None:
        st.success("Good", icon="✅")
                                     
    #        model, class_names = load_model_and_class_names()
    #        img = Image.open(file)
    #        img_placeholder.image(img, width=299)
    #    submit = submit_placeholder.button("Lancer la détection de race")

    #if submit:
    #    with st.spinner('Résultat en attente...'):    
    #        submit_placeholder.empty()
    #        img_tensor = load_image(img)
    #        res = get_prediction(model=model, img=img_tensor, class_names=class_names)
    #        success.success("Pour le modèle il s'agit d'un {}".format(res))

if __name__ == '__main__':
    main()