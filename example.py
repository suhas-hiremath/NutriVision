import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import requests
import os

model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

def fetch_calories(prediction):
    try:
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={prediction}&search_simple=1&action=process&json=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['products']:
                product = data['products'][0]
                calories = product.get('nutriments', {}).get('energy-kcal_100g', "Data not available")
                return f"{calories} kcal" if calories != "Data not available" else calories
            else:
                return "No data available"
        else:
            return "Error fetching data"
    except Exception as e:
        return "Can't fetch calories"


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int("".join(str(x) for x in y_class))
    res = labels[y]
    return res.capitalize()


def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    # Add camera input
    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/camera_image.jpg'
        os.makedirs('./upload_images/', exist_ok=True)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = processed_img(save_image_path)
        st.success("**Predicted: " + result + '**')
        cal = fetch_calories(result)
        if cal:
            st.warning('**Calories: ' + cal + '(100 grams)**')


run()
