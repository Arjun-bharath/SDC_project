import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Load images and masks
def load_data(original_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []

    original_files = sorted(os.listdir(original_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for orig, mask in zip(original_files, mask_files):
        img = cv2.imread(os.path.join(original_dir, orig), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(os.path.join(mask_dir, mask), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, img_size) / 255.0
        mask_img = cv2.resize(mask_img, img_size) / 255.0

        images.append(img)
        masks.append(mask_img)

    return np.array(images).reshape(-1, img_size[0], img_size[1], 1), np.array(masks).reshape(-1, img_size[0], img_size[1], 1)

# Paths
original_img_dir = "/Users/ajaykarthick/Downloads/final_draft/images"
mask_img_dir = "/Users/ajaykarthick/Downloads/final_draft/masks"

# Load and split data
X, Y = load_data(original_img_dir, mask_img_dir)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256, 256, 1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2,2)),
        tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2,2)),
        tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Train model
    model = build_model()
    model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

    # Save model
    model.save("tumor_segmentation_model.h5")

    # Streamlit App
    st.title("Brain Tumor Segmentation")
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256)) / 255.0
        img = img.reshape(1, 256, 256, 1)

        model = tf.keras.models.load_model("tumor_segmentation_model.h5")
        prediction = model.predict(img)[0]

        st.image(prediction, caption="Segmented Tumor", use_column_width=True)
if '__name__'=="__main__":
    main()
