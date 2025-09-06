import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# ------------------
# Page Config
# ------------------
st.set_page_config(
    page_title="🍽️ MealSnap AI – Nutrition Estimator",
    page_icon="🍎",
    layout="wide"
)


# --------------------------
# Background
# --------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bg.png")   # your background image


# --------------------------
# Edamam API Keys
# --------------------------
APP_ID = "e9755876"
APP_KEY = "49fa98e3b702be8cf3bd51d5ffdc5a67"


# --------------------------
# Nutrition Function
# --------------------------
def get_nutrition(food_item):
    """Fetch nutrition info from Edamam API and display results"""
    url = f"https://api.edamam.com/api/nutrition-data?app_id={APP_ID}&app_key={APP_KEY}&ingr={food_item}"
    response = requests.get(url).json()

    try:
        nutrients = response["ingredients"][0]["parsed"][0]["nutrients"]

        calories = nutrients["ENERC_KCAL"]["quantity"]
        protein = nutrients["PROCNT"]["quantity"]
        carbs = nutrients["CHOCDF"]["quantity"]
        fat = nutrients["FAT"]["quantity"]

        # Display results
        st.subheader(f"Nutrition for: {food_item}")
        st.write(f"**Calories:** {calories:.2f} kcal")
        st.write(f"**Protein:** {protein:.2f} g")
        st.write(f"**Carbs:** {carbs:.2f} g")
        st.write(f"**Fat:** {fat:.2f} g")

        # Visualization
        data = {
            "Nutrient": ["Calories", "Protein", "Carbs", "Fat"],
            "Amount": [calories, protein, carbs, fat]
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots()
        ax.bar(df["Nutrient"], df["Amount"], color=["orange", "green", "blue", "red"])
        ax.set_ylabel("Amount")
        ax.set_title("Nutritional Breakdown")
        st.pyplot(fig)

    except Exception:
        st.error("⚠️ Could not fetch nutrition info. Try rephrasing (e.g., '100g chicken').")


# --------------------------
# Load YOLO Model
# --------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # replace with custom food-trained weights if available

model = load_model()


# --------------------------
# Streamlit App
# --------------------------
st.title("🍽️ MealSnap AI – Nutrition Estimator")

option = st.radio("Choose Input Method:", ["Search Bar", "Camera"])


# --- Search Bar ---
if option == "Search Bar":
    food_item = st.text_input("Enter food with quantity (e.g., '1 cup rice', '100g chicken')")
    if st.button("Get Nutrition Info"):
        if food_item.strip() == "":
            st.warning("⚠️ Please enter a food item with quantity.")
        else:
            get_nutrition(food_item)


# --- Camera Input ---
elif option == "Camera":
    uploaded_img = st.camera_input("📸 Take a picture of your food")

    if uploaded_img is not None:
        # Convert to OpenCV image
        image = Image.open(uploaded_img)
        img_array = np.array(image)

        # YOLO Prediction
        results = model.predict(img_array, conf=0.5)
        detected_items = set()
        for r in results:
            for c in r.boxes.cls:
                detected_items.add(model.names[int(c)])

        if detected_items:
            st.success(f"Detected: {', '.join(detected_items)}")

            # Add "1" quantity by default
            for item in detected_items:
                food_with_qty = f"1 {item}"   # <-- FIX HERE
                get_nutrition(food_with_qty)
                break
        else:
            st.warning("⚠️ No recognizable food detected. Try again.")

# Credits
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; font-size: 14px; color: grey;">
        © Made by <b>Pranjal Purohit</b>
    </div>
    """,
    unsafe_allow_html=True
)
