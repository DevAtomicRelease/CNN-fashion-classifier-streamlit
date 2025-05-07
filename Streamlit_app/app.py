import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageOps
import os

# Loading  the trained model
model = load_model('fashion_mnist_cnn_model.h5')

# Classes names for Fashion-MNIST to classify the pridiction
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title('🧥👚 Fashion-MNIST Image Classifier')

st.write("""
Upload a grayscale image (preferably 28x28 pixels).  
The model will predict the clothing item category.
""")
#uploding the image to the modal for testing
uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png", "svg", "webp"])

# Creating a path to store corrected freture in the same folder as app.py file
if not os.path.exists('corrections'):
    os.makedirs('corrections')

if uploaded_file is not None:
    st.subheader("Uploaded Image:")
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Preprocessing the image
    img = ImageOps.invert(image)        
    img = img.resize((28, 28))      
    img_array = np.array(img).astype('float32') / 255.0  
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.write(f"### Predicted Class: **{predicted_class}**")
    st.write(f"### Confidence Level: **{confidence:.2f}%**")

    # User feedback option to re-train modal afterwards
    if 'correcting' not in st.session_state:
        st.session_state['correcting'] = False

    if st.button('Wrong Prediction?'):
        st.session_state['correcting'] = True

    if st.session_state['correcting']:
        correct_label = st.selectbox('Select the correct label:', class_names)

        if st.button('Submit Correction'):
            # Save corrected image with its labal
            correction_path = os.path.join('corrections', f'{correct_label}_{uploaded_file.name}')
            image.save(correction_path)
            st.success(f'Correction saved successfully as {correction_path}')

            # Save correction into CSV
            correction_data = pd.DataFrame([[uploaded_file.name, correct_label]], columns=['Filename', 'Correct_Label'])
            csv_path = os.path.join('corrections', 'corrections.csv')
            # Appending the correct data into the csv file
            if os.path.exists(csv_path):
                correction_data.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                correction_data.to_csv(csv_path, index=False)

            # Reset correction session
            st.session_state['correcting'] = False
