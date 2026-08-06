import streamlit as st
import requests
import matplotlib.pyplot as plt
import base64
from google import genai
import json
import os
from PIL import Image


# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="MealSnap AI – Smart Nutrition Estimator",
    page_icon="🍽️",
    layout="wide"
)


# ==============================================================
# BACKGROUND
# ==============================================================

def add_bg_from_local(image_file):
    if os.path.exists(image_file):
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
            unsafe_allow_html=True,
        )


add_bg_from_local("bg.png")


# ==============================================================
# API KEYS
# ==============================================================

# Never hardcode API keys here.
# Store all credentials in Streamlit Cloud → Settings → Secrets.

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    EDAMAM_APP_ID = st.secrets["EDAMAM_APP_ID"]
    EDAMAM_APP_KEY = st.secrets["EDAMAM_APP_KEY"]

except KeyError as e:
    st.error(
        f"❌ Missing secret: {e}\n\n"
        "Add GEMINI_API_KEY, EDAMAM_APP_ID and EDAMAM_APP_KEY "
        "to Streamlit Secrets."
    )
    st.stop()


# ==============================================================
# GEMINI CONFIGURATION
# ==============================================================

GEMINI_MODEL = "models/gemini-3.5-flash"


@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ==============================================================
# HELPER — CLEAN GEMINI JSON
# ==============================================================

def clean_json_response(raw_text):
    """
    Removes possible markdown fences from Gemini output.
    """

    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```JSON", "")
        raw_text = raw_text.replace("```", "")
        raw_text = raw_text.strip()

    return raw_text


# ==============================================================
# FOOD DETECTION — GEMINI VISION
# ==============================================================

def detect_food_with_gemini(image: Image.Image):
    """
    Detect food items from an uploaded/captured image.

    Returns:
        [
            {
                "item": str,
                "count": int,
                "serving": str
            }
        ]
    """

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    client = get_gemini_client()

    prompt = """
You are a professional food-recognition and nutrition assistant.

Analyze the supplied image and identify ALL visible food items.

For every food item:

1. Give the most specific food name possible.
   Examples:
   - "scrambled eggs" instead of "eggs"
   - "basmati rice" instead of "rice"
   - "dal makhani" instead of "dal"

2. Count individual pieces or portions when possible.
   Examples:
   - 2 chapatis
   - 3 samosas
   - 4 idlis
   - 1 bowl of dal

3. Estimate a realistic serving quantity that can be sent
   to a nutrition API.

Examples of serving strings:
- "2 medium apples"
- "1 cup cooked basmati rice"
- "100g paneer"
- "2 medium chapatis"
- "1 slice vegetable pizza"

IMPORTANT RULES:

- Include all clearly visible food items.
- Count multiple pieces of the same food.
- Ignore plates, cutlery, glasses, hands and background objects.
- Do not invent food that cannot reasonably be seen.
- Return [] if no food is visible.
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT include explanations.

Required JSON format:

[
  {
    "item": "food name",
    "count": 2,
    "serving": "2 medium food items"
  }
]
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image]
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    raw = clean_json_response(response.text)

    detected_foods = json.loads(raw)

    if not isinstance(detected_foods, list):
        raise ValueError("Gemini response was not a JSON list.")

    cleaned_foods = []

    for food in detected_foods:

        if not isinstance(food, dict):
            continue

        item = food.get("item")
        serving = food.get("serving")
        count = food.get("count", 1)

        if not item or not serving:
            continue

        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 1

        cleaned_foods.append(
            {
                "item": str(item),
                "count": max(count, 1),
                "serving": str(serving)
            }
        )

    return cleaned_foods


# ==============================================================
# NUTRITION — EDAMAM API
# ==============================================================

def get_nutrition(serving_query: str, display_name: str = None):

    if display_name is None:
        display_name = serving_query

    url = "https://api.edamam.com/api/nutrition-data"

    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "ingr": serving_query,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        resp = response.json()

        ingredients = resp.get("ingredients", [])

        if not ingredients:
            st.warning(
                f"⚠️ Nutrition information couldn't be found for "
                f"**{serving_query}**."
            )
            return None

        parsed = ingredients[0].get("parsed", [])

        if not parsed:
            st.warning(
                f"⚠️ Edamam couldn't understand: "
                f"**{serving_query}**."
            )
            return None

        nutrients = parsed[0].get("nutrients", {})

        calories = nutrients.get(
            "ENERC_KCAL", {}
        ).get("quantity", 0)

        protein = nutrients.get(
            "PROCNT", {}
        ).get("quantity", 0)

        carbs = nutrients.get(
            "CHOCDF", {}
        ).get("quantity", 0)

        fat = nutrients.get(
            "FAT", {}
        ).get("quantity", 0)

        fiber = nutrients.get(
            "FIBTG", {}
        ).get("quantity", 0)

        st.markdown(
            f"""
            <div style="
                background-color:rgba(0,0,0,0.72);
                padding:18px 22px;
                border-radius:12px;
                color:white;
                margin-bottom:16px;
                border-left:4px solid #f0a500;
            ">

                <h3 style="margin:0 0 10px 0;">
                    🍴 {display_name}
                </h3>

                <table style="width:100%; font-size:15px;">

                    <tr>
                        <td>🔥 <b>Calories</b></td>
                        <td><b>{calories:.1f} kcal</b></td>
                    </tr>

                    <tr>
                        <td>💪 <b>Protein</b></td>
                        <td>{protein:.1f} g</td>
                    </tr>

                    <tr>
                        <td>🌾 <b>Carbs</b></td>
                        <td>{carbs:.1f} g</td>
                    </tr>

                    <tr>
                        <td>🧈 <b>Fat</b></td>
                        <td>{fat:.1f} g</td>
                    </tr>

                    <tr>
                        <td>🌿 <b>Fiber</b></td>
                        <td>{fiber:.1f} g</td>
                    </tr>

                </table>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------
        # NUTRITION CHART
        # ------------------------------------------------------

        fig, ax = plt.subplots(figsize=(5, 2.5))

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.bar(
            [
                "Calories",
                "Protein",
                "Carbs",
                "Fat",
                "Fiber"
            ],
            [
                calories,
                protein,
                carbs,
                fat,
                fiber
            ],
            color=[
                "#f0a500",
                "#4caf50",
                "#2196f3",
                "#f44336",
                "#9c27b0"
            ],
            edgecolor="white",
            linewidth=0.6,
        )

        ax.set_ylabel(
            "Amount",
            color="white"
        )

        ax.set_title(
            display_name,
            color="white",
            fontsize=10
        )

        ax.tick_params(
            colors="white",
            labelsize=8
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        st.pyplot(fig)

        plt.close(fig)

        return {
            "item": display_name,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "fiber": fiber,
        }

    except requests.exceptions.Timeout:

        st.error(
            "❌ Nutrition API timed out. Please try again."
        )

        return None

    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ Nutrition API error: {e}"
        )

        return None

    except (KeyError, IndexError, TypeError):

        st.warning(
            f"⚠️ Edamam couldn't find nutrition for: "
            f"**{serving_query}**."
        )

        return None

    except Exception as e:

        st.error(
            f"❌ Unexpected nutrition error: {e}"
        )

        return None


# ==============================================================
# MEAL SUMMARY
# ==============================================================

def show_meal_summary(nutrition_list: list):

    valid = [
        nutrition
        for nutrition in nutrition_list
        if nutrition
    ]

    if not valid:
        return None

    total = {
        key: sum(
            nutrition.get(key, 0)
            for nutrition in valid
        )
        for key in (
            "calories",
            "protein",
            "carbs",
            "fat",
            "fiber"
        )
    }

    st.markdown(
        f"""
        <div style="
            background-color:rgba(0,100,0,0.75);
            padding:20px 24px;
            border-radius:12px;
            color:white;
            margin:20px 0;
            border-left:4px solid #00e676;
        ">

            <h2 style="margin:0 0 12px 0;">
                📊 Total Meal Summary
            </h2>

            <table style="width:100%; font-size:16px;">

                <tr>
                    <td>🔥 <b>Total Calories</b></td>
                    <td>
                        <b>{total['calories']:.1f} kcal</b>
                    </td>
                </tr>

                <tr>
                    <td>💪 <b>Total Protein</b></td>
                    <td>{total['protein']:.1f} g</td>
                </tr>

                <tr>
                    <td>🌾 <b>Total Carbs</b></td>
                    <td>{total['carbs']:.1f} g</td>
                </tr>

                <tr>
                    <td>🧈 <b>Total Fat</b></td>
                    <td>{total['fat']:.1f} g</td>
                </tr>

                <tr>
                    <td>🌿 <b>Total Fiber</b></td>
                    <td>{total['fiber']:.1f} g</td>
                </tr>

            </table>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------
    # MACRO PIE CHART
    # ----------------------------------------------------------

    labels = [
        "Protein",
        "Carbs",
        "Fat"
    ]

    sizes = [
        total["protein"],
        total["carbs"],
        total["fat"]
    ]

    colors = [
        "#4caf50",
        "#2196f3",
        "#f44336"
    ]

    if sum(sizes) > 0:

        fig, ax = plt.subplots(
            figsize=(4, 4)
        )

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        _, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={
                "color": "white",
                "fontsize": 10
            },
        )

        for text in autotexts:
            text.set_color("white")

        ax.set_title(
            "Macro Distribution",
            color="white",
            fontsize=12
        )

        st.pyplot(fig)

        plt.close(fig)

    return total


# ==============================================================
# PERSONALIZED SUGGESTIONS — GEMINI
# ==============================================================

def suggest_additions(total: dict, food_context: str):

    # ----------------------------------------------------------
    # IDENTIFY NUTRITIONAL GAPS
    # ----------------------------------------------------------

    gaps = []

    if total.get("calories", 0) < 300:
        gaps.append(
            f"Calories are low "
            f"({total.get('calories', 0):.0f} kcal; "
            f"meal target approximately 300+ kcal)."
        )

    if total.get("protein", 0) < 20:
        gaps.append(
            f"Protein is low "
            f"({total.get('protein', 0):.1f} g; "
            f"meal target approximately 20+ g)."
        )

    if total.get("fat", 0) < 10:
        gaps.append(
            f"Healthy fats are low "
            f"({total.get('fat', 0):.1f} g; "
            f"meal target approximately 10+ g)."
        )

    if total.get("carbs", 0) < 30:
        gaps.append(
            f"Carbohydrates are low "
            f"({total.get('carbs', 0):.1f} g; "
            f"meal target approximately 30+ g)."
        )

    if total.get("fiber", 0) < 5:
        gaps.append(
            f"Fiber is low "
            f"({total.get('fiber', 0):.1f} g; "
            f"meal target approximately 5+ g)."
        )

    if not gaps:

        st.success(
            "✅ This meal appears reasonably balanced "
            "for the nutrients being measured."
        )

        return

    gap_text = "\n".join(
        f"- {gap}"
        for gap in gaps
    )

    # ----------------------------------------------------------
    # GEMINI PROMPT
    # ----------------------------------------------------------

    prompt = f"""
You are a nutrition assistant.

The user ate:

{food_context}

Based on the available nutrition API data, these nutritional
gaps were detected:

{gap_text}

Suggest EXACTLY 3 to 5 practical food additions.

IMPORTANT:

1. Suggestions should make sense with the existing meal.

2. Respect the apparent food category.

Examples:

- If the meal contains only fruits, recommend fruit-compatible
  foods such as yogurt, nuts, seeds or other appropriate foods.

- If the meal appears vegetarian, do not randomly recommend meat.

- If the meal appears vegan, prefer plant-based additions.

- For Indian meals, prefer practical additions that naturally
  complement the meal where appropriate.

3. Prioritize the nutritional gaps listed above.

4. Keep every reason short.

5. Do not claim that these targets are personalized medical
   requirements.

6. Return between 3 and 5 suggestions.

Return ONLY valid JSON.

Do not include markdown fences.

Required format:

[
  {{
    "suggestion": "Add a small bowl of Greek yogurt",
    "reason": "Adds protein and complements the existing meal.",
    "emoji": "🥛"
  }},
  {{
    "suggestion": "Add a handful of almonds",
    "reason": "Adds healthy fats, protein and fiber.",
    "emoji": "🌰"
  }},
  {{
    "suggestion": "Add chia seeds",
    "reason": "Provides fiber and healthy fats.",
    "emoji": "🌱"
  }}
]
"""

    try:

        client = get_gemini_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        raw_text = clean_json_response(
            response.text
        )

        suggestions = json.loads(
            raw_text
        )

        if not isinstance(suggestions, list):
            raise ValueError(
                "Gemini response was not a JSON list."
            )

        # Maximum 5 suggestions
        suggestions = suggestions[:5]

        # Validate suggestions
        valid_suggestions = []

        for suggestion in suggestions:

            if not isinstance(suggestion, dict):
                continue

            title = suggestion.get(
                "suggestion"
            )

            reason = suggestion.get(
                "reason"
            )

            if not title or not reason:
                continue

            valid_suggestions.append(
                {
                    "suggestion": str(title),
                    "reason": str(reason),
                    "emoji": str(
                        suggestion.get(
                            "emoji",
                            "🍴"
                        )
                    )
                }
            )

        if not valid_suggestions:
            raise ValueError(
                "Gemini didn't return usable suggestions."
            )

        # ------------------------------------------------------
        # DISPLAY SUGGESTIONS
        # ------------------------------------------------------

        st.markdown(
            """
            <div style="
                background-color:rgba(0,0,0,0.70);
                padding:16px 20px;
                border-radius:12px;
                color:white;
                margin-bottom:10px;
                border-left:4px solid #ffeb3b;
            ">

                <h3 style="margin:0;">
                    💡 Personalized Suggestions
                </h3>

            </div>
            """,
            unsafe_allow_html=True,
        )

        for suggestion in valid_suggestions:

            emoji = suggestion["emoji"]
            title = suggestion["suggestion"]
            reason = suggestion["reason"]

            st.markdown(
                f"""
                <div style="
                    margin:8px 0;
                    padding:10px 14px;
                    background:rgba(0,0,0,0.65);
                    border-radius:8px;
                    color:white;
                ">

                    <span style="font-size:18px;">
                        {emoji}
                    </span>

                    <b>
                        {title}
                    </b>

                    <br>

                    <span style="
                        font-size:13px;
                        color:#ddd;
                    ">
                        {reason}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

    except json.JSONDecodeError:

        st.warning(
            "⚠️ Gemini returned an unexpected response. "
            "Please try again."
        )

    except Exception as e:

        error_message = str(e)

        if (
            "429" in error_message
            or "quota" in error_message.lower()
            or "resource_exhausted" in error_message.lower()
        ):

            st.warning(
                "⏳ Gemini API rate limit or quota reached. "
                "Please wait and try again."
            )

        elif (
            "api key" in error_message.lower()
            or "api_key" in error_message.lower()
            or "credentials" in error_message.lower()
        ):

            st.error(
                "❌ Gemini authentication failed. "
                "Check GEMINI_API_KEY in Streamlit Secrets."
            )

        else:

            st.warning(
                f"⚠️ Could not generate personalized "
                f"suggestions: {error_message}"
            )


# ==============================================================
# APP LAYOUT
# ==============================================================

st.title(
    "🍽️ MealSnap AI – Smart Nutrition Estimator"
)

st.caption(
    "Snap your food, get the nutrition!"
)

option = st.radio(
    "Choose Input Method:",
    [
        "🔍 Search Bar",
        "📸 Camera / Upload"
    ]
)


# ==============================================================
# SEARCH BAR
# ==============================================================

if option == "🔍 Search Bar":

    food_item = st.text_input(
        "Enter food with quantity",
        placeholder=(
            "e.g. '2 medium apples', "
            "'1 cup basmati rice', "
            "'100g paneer'"
        ),
    )

    if st.button(
        "Get Nutrition Info",
        type="primary"
    ):

        if not food_item.strip():

            st.warning(
                "⚠️ Please enter a food item with quantity."
            )

        else:

            nutrition = get_nutrition(
                food_item
            )

            if nutrition:

                suggest_additions(
                    nutrition,
                    food_context=food_item
                )


# ==============================================================
# CAMERA / UPLOAD
# ==============================================================

elif option == "📸 Camera / Upload":

    input_mode = st.radio(
        "Input type:",
        [
            "Camera",
            "Upload Image"
        ],
        horizontal=True
    )

    uploaded_img = None

    if input_mode == "Camera":

        uploaded_img = st.camera_input(
            "📸 Take a picture of your food"
        )

    else:

        uploaded_img = st.file_uploader(
            "Upload a food image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

    if uploaded_img is not None:

        try:

            image = Image.open(
                uploaded_img
            )

            st.image(
                image,
                caption="Your food image",
                width=400
            )

        except Exception:

            st.error(
                "❌ Could not open this image."
            )

            st.stop()

        # ------------------------------------------------------
        # FOOD DETECTION
        # ------------------------------------------------------

        with st.spinner(
            "🤖 MealSnap AI is analysing your food..."
        ):

            try:

                detected_foods = (
                    detect_food_with_gemini(
                        image
                    )
                )

            except json.JSONDecodeError:

                st.error(
                    "❌ Gemini returned an unexpected "
                    "food-detection response. Please try again."
                )

                detected_foods = []

            except Exception as e:

                error_message = str(e)

                if (
                    "429" in error_message
                    or "quota" in error_message.lower()
                    or "resource_exhausted"
                    in error_message.lower()
                ):

                    st.error(
                        "⏳ Gemini API rate limit or quota "
                        "reached. Please wait and retry."
                    )

                elif (
                    "api key" in error_message.lower()
                    or "api_key" in error_message.lower()
                    or "credentials"
                    in error_message.lower()
                ):

                    st.error(
                        "❌ Invalid Gemini API key. "
                        "Check Streamlit Secrets."
                    )

                else:

                    st.error(
                        f"❌ Detection error: "
                        f"{error_message}"
                    )

                detected_foods = []

        # ------------------------------------------------------
        # RESULTS
        # ------------------------------------------------------

        if not detected_foods:

            st.warning(
                """
⚠️ No food detected.

Tips:

- Ensure the food is clearly visible.
- Use good lighting.
- Avoid blurry images.
- Capture the whole plate.
- Try taking the image from above.
                """
            )

        else:

            st.markdown(
                "### 🔍 Detected Food Items"
            )

            for food in detected_foods:

                st.markdown(
                    f"""
                    <div style="
                        background-color:rgba(0,0,0,0.60);
                        padding:10px 16px;
                        border-radius:8px;
                        color:white;
                        margin:5px 0;
                        border-left:3px solid #f0a500;
                    ">

                        🍴 <b>
                            {food['item']}
                        </b>

                        &nbsp;|&nbsp;

                        Qty:
                        <b>
                            {food['count']}
                        </b>

                        &nbsp;|&nbsp;

                        Serving:
                        <i>
                            {food['serving']}
                        </i>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            st.markdown(
                "### 🧪 Nutrition Breakdown (per item)"
            )

            nutrition_results = []

            for food in detected_foods:

                result = get_nutrition(
                    serving_query=food["serving"],
                    display_name=(
                        f"{food['count']}× "
                        f"{food['item']}"
                    ),
                )

                if result:
                    nutrition_results.append(
                        result
                    )

            # --------------------------------------------------
            # TOTAL
            # --------------------------------------------------

            if nutrition_results:

                st.markdown("---")

                total = show_meal_summary(
                    nutrition_results
                )

                if total:

                    food_context = ", ".join(
                        f"{food['count']} "
                        f"{food['item']}"
                        for food
                        in detected_foods
                    )

                    suggest_additions(
                        total,
                        food_context=food_context
                    )

            else:

                st.warning(
                    "⚠️ Nutrition information couldn't "
                    "be calculated for the detected foods."
                )


# ==============================================================
# DISCLAIMER
# ==============================================================

st.markdown("---")

st.caption(
    "Nutrition values and AI food recognition are estimates. "
    "They should not be treated as medical or dietary advice."
)


# ==============================================================
# CREDITS
# ==============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:30px;
        font-size:13px;
        color:grey;
    ">
        © Made by <b>Pranjal Purohit</b>
    </div>
    """,
    unsafe_allow_html=True,
)
