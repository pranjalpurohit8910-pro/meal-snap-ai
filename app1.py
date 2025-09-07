import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np


# Page Title
st.set_page_config(
    page_title="MealSnap AI – Smart Nutrition Estimator",
    page_icon="🍽️",
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

add_bg_from_local("bg.png")


# --------------------------
# Edamam API Keys
# --------------------------
APP_ID = "e9755876"
APP_KEY = "49fa98e3b702be8cf3bd51d5ffdc5a67"


# --------------------------
# Nutrition Function
# --------------------------
def get_nutrition(food_item):
    url = f"https://api.edamam.com/api/nutrition-data?app_id={APP_ID}&app_key={APP_KEY}&ingr={food_item}"
    response = requests.get(url).json()

    try:
        nutrients = response["ingredients"][0]["parsed"][0]["nutrients"]

        calories = nutrients["ENERC_KCAL"]["quantity"]
        protein = nutrients["PROCNT"]["quantity"]
        carbs = nutrients["CHOCDF"]["quantity"]
        fat = nutrients["FAT"]["quantity"]

        # Display results inside black box
        st.markdown(
            f"""
            <div style="background-color: rgba(0, 0, 0, 0.7); 
                        padding: 20px; 
                        border-radius: 10px; 
                        color: white;
                        margin-bottom: 20px;">
                <h2>Nutrition for: {food_item}</h2>
                <p><b>Calories:</b> {calories:.2f} kcal</p>
                <p><b>Protein:</b> {protein:.2f} g</p>
                <p><b>Carbs:</b> {carbs:.2f} g</p>
                <p><b>Fat:</b> {fat:.2f} g</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        return {
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat
        }

    except Exception:
        st.error("⚠️ Could not fetch nutrition info. Try rephrasing (e.g., '100g chicken').")
        return None


# --------------------------
# Suggest Additions Function
# --------------------------
def suggest_additions(nutrition):
    suggestions = []

    if nutrition["protein"] < 20:
        suggestions.append("🍳 Add a boiled egg to boost protein.")
    if nutrition["fat"] < 10:
        suggestions.append("🥑 Add some avocado for healthy fats.")
    if nutrition["carbs"] < 30:
        suggestions.append("🍞 Add a slice of whole grain bread for more carbs.")
    if nutrition["calories"] < 300:
        suggestions.append("🥗 Add a small salad to increase calories.")

    if suggestions:
        st.markdown(
            """
            <div style="background-color: rgba(0, 0, 0, 0.7); 
                        padding: 15px; 
                        border-radius: 10px; 
                        color: white;
                        margin-bottom: 20px;">
                <h3>💡 Suggestions to improve your meal:</h3>
            """,
            unsafe_allow_html=True
        )
        for s in suggestions:
            st.markdown(f"- {s}", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Your meal looks balanced!")


# --------------------------
# Load YOLO Model
# --------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Replace with custom food-trained weights if available

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
            nutrition = get_nutrition(food_item)
            if nutrition:
                suggest_additions(nutrition)


# --- Camera Input ---
elif option == "Camera":
    uploaded_img = st.camera_input("📸 Take a picture of your food")

    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        img_array = np.array(image)

        results = model.predict(img_array, conf=0.5)
        detected_items = set()
        for r in results:
            for c in r.boxes.cls:
                detected_items.add(model.names[int(c)])

        if detected_items:
            st.success(f"Detected: {', '.join(detected_items)}")

            for item in detected_items:
                food_with_qty = f"1 {item}"
                get_nutrition(food_with_qty)
                break
        else:
            st.warning("⚠️ No recognizable food detected. Try again.")


# --------------------------
# Credits / Watermark
# --------------------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; font-size: 14px; color: grey;">
        © Made by <b>Pranjal Purohit</b>
    </div>
    """,
    unsafe_allow_html=True
)
