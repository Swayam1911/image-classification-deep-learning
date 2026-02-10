import streamlit as st
from PIL import Image
from src.preprocess import preprocess_image
from src.predict import predict

st.set_page_config(page_title="Flower Classifier 🌸")

st.title("🌸 Flower Image Classification")
st.write("Upload a flower image to get prediction")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess_image(image)
    label, confidence = predict(img)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")


