import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sports_100_model_finetuned.h5")

model = load_model()

# IMPORTANT: class names in correct order
class_names = sorted([
'air hockey','ampute football','archery','arm wrestling','axe throwing',
'balance beam','barell racing','baseball','basketball','baton twirling',
'bike polo','billiards','bmx','bobsled','bowling','boxing','bull riding',
'bungee jumping','canoe slamon','cheerleading','chuckwagon racing','cricket',
'croquet','curling','disc golf','fencing','field hockey',
'figure skating men','figure skating pairs','figure skating women',
'fly fishing','football','formula 1 racing','frisbee','gaga','giant slalom',
'golf','hammer throw','hang gliding','harness racing','high jump','hockey',
'horse jumping','horse racing','horseshoe pitching','hurdles',
'hydroplane racing','ice climbing','ice yachting','jai alai','javelin',
'jousting','judo','lacrosse','log rolling','luge','motorcycle racing',
'mushing','nascar racing','olympic wrestling','parallel bar',
'pole climbing','pole dancing','pole vault','polo','pommel horse',
'rings','rock climbing','roller derby','rollerblade racing','rowing',
'rugby','sailboat racing','shot put','shuffleboard','sidecar racing',
'ski jumping','sky surfing','skydiving','snow boarding',
'snowmobile racing','speed skating','steer wrestling','sumo wrestling',
'surfing','swimming','table tennis','tennis','track bicycle','trapeze',
'tug of war','ultimate','uneven bars','volleyball','water cycling',
'water polo','weightlifting','wheelchair basketball',
'wheelchair racing','wingsuit flying'
])

st.title("🏆 Sports Image Classifier (100 Classes)")
st.write("Upload an image to classify the sport.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
