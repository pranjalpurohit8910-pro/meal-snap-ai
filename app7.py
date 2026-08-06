import streamlit as st
import requests
import matplotlib.pyplot as plt
import base64
from google import genai
import json
import os
import time
import html
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
        try:
            with open(image_file, "rb") as file:
                encoded_string = base64.b64encode(file.read()).decode()

            css = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
"""
            st.markdown(css, unsafe_allow_html=True)

        except Exception as e:
            print(f"Background image error: {e}")


add_bg_from_local("bg.png")


# ==============================================================
# STREAMLIT SECRETS
# ==============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    EDAMAM_APP_ID = st.secrets["EDAMAM_APP_ID"]
    EDAMAM_APP_KEY = st.secrets["EDAMAM_APP_KEY"]

except KeyError as e:
    st.error(
        "❌ Required API credentials are missing from Streamlit Secrets."
    )
    print(f"Missing Streamlit secret: {e}")
    st.stop()


# ==============================================================
# GEMINI CONFIGURATION
# ==============================================================

# Models confirmed available to your API key.
# If one model returns 503/high demand, the next one is tried.

GEMINI_MODELS = [
    "models/gemini-3.5-flash",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash-lite",
]


@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ==============================================================
# GEMINI FALLBACK
# ==============================================================

def generate_with_fallback(contents):
    """
    Try Gemini models in sequence.

    Temporary 429/503/capacity errors cause the app to
    automatically try another available model.
    """

    client = get_gemini_client()

    last_error = None

    for model_name in GEMINI_MODELS:

        # Two attempts per model for temporary service issues.
        for attempt in range(2):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )

                if response and response.text:
                    if response.text.strip():
                        return response

                last_error = RuntimeError(
                    f"{model_name} returned an empty response."
                )

            except Exception as e:

                last_error = e
                error_text = str(e).lower()

                temporary_error = (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "high demand" in error_text
                    or "temporarily" in error_text
                    or "429" in error_text
                    or "resource_exhausted" in error_text
                    or "resource exhausted" in error_text
                    or "rate limit" in error_text
                )

                if temporary_error:
                    print(
                        f"Temporary Gemini error using "
                        f"{model_name}, attempt {attempt + 1}: {e}"
                    )

                    # Small delay before retrying.
                    if attempt == 0:
                        time.sleep(1)

                    continue

                # Authentication/model-format/programming errors
                # should not silently move through all models.
                raise

        # Current model failed twice.
        # Continue with next fallback model.

    print(f"All Gemini models failed: {last_error}")

    raise RuntimeError(
        "AI service is temporarily unavailable."
    )


# ==============================================================
# CLEAN GEMINI JSON
# ==============================================================

def clean_json_response(raw_text):

    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    raw_text = raw_text.strip()

    # Remove markdown code fences if Gemini adds them.
    if raw_text.startswith("```"):

        lines = raw_text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_text = "\n".join(lines).strip()

    # Remove accidental "json" prefix.
    if raw_text.lower().startswith("json"):
        raw_text = raw_text[4:].strip()

    return raw_text


# ==============================================================
# FOOD DETECTION — GEMINI VISION
# ==============================================================

def detect_food_with_gemini(image: Image.Image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    prompt = """
You are a food recognition assistant.

Analyze the supplied food image.

Identify ALL clearly visible food items.

For each item:

1. Give the most specific reasonable food name.
2. Count visible pieces or portions when possible.
3. Estimate a realistic serving quantity suitable for a nutrition API.

Examples:

2 chapatis:
{
  "item": "chapati",
  "count": 2,
  "serving": "2 medium chapatis"
}

1 bowl dal:
{
  "item": "dal",
  "count": 1,
  "serving": "1 cup cooked dal"
}

IMPORTANT RULES:

- Include all clearly visible foods.
- Count multiple pieces of the same food.
- Do not include plates, spoons, glasses, hands or background objects.
- Do not invent foods that cannot reasonably be identified.
- If uncertain between similar foods, use the most reasonable general name.
- Return [] when no food is visible.
- Return ONLY valid JSON.
- Do NOT return Markdown.
- Do NOT provide explanations.

Required format:

[
  {
    "item": "food name",
    "count": 1,
    "serving": "realistic serving"
  }
]
"""

    response = generate_with_fallback(
        [prompt, image]
    )

    raw = clean_json_response(
        response.text
    )

    detected_foods = json.loads(raw)

    if not isinstance(detected_foods, list):
        raise ValueError(
            "Food detection response was not a list."
        )

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
                "item": str(item).strip(),
                "count": max(count, 1),
                "serving": str(serving).strip()
            }
        )

    return cleaned_foods


# ==============================================================
# NUTRITION — EDAMAM
# ==============================================================

def get_nutrition(serving_query, display_name=None):

    if display_name is None:
        display_name = serving_query

    url = "https://api.edamam.com/api/nutrition-data"

    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "ingr": serving_query
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        ingredients = data.get(
            "ingredients",
            []
        )

        if not ingredients:

            st.warning(
                f"⚠️ Nutrition information couldn't be found "
                f"for **{serving_query}**."
            )

            return None

        parsed = ingredients[0].get(
            "parsed",
            []
        )

        if not parsed:

            st.warning(
                f"⚠️ Nutrition information couldn't be understood "
                f"for **{serving_query}**."
            )

            return None

        nutrients = parsed[0].get(
            "nutrients",
            {}
        )

        calories = nutrients.get(
            "ENERC_KCAL", {}
        ).get("quantity", 0) or 0

        protein = nutrients.get(
            "PROCNT", {}
        ).get("quantity", 0) or 0

        carbs = nutrients.get(
            "CHOCDF", {}
        ).get("quantity", 0) or 0

        fat = nutrients.get(
            "FAT", {}
        ).get("quantity", 0) or 0

        fiber = nutrients.get(
            "FIBTG", {}
        ).get("quantity", 0) or 0

        # Escape user/API-generated text before inserting into HTML.
        safe_display_name = html.escape(
            str(display_name)
        )

        # IMPORTANT:
        # HTML starts at the beginning of each line.
        # This prevents Streamlit Markdown from treating it as code.

        nutrition_html = f"""<div style="background-color:rgba(0,0,0,0.72);padding:18px 22px;border-radius:12px;color:white;margin-bottom:16px;border-left:4px solid #f0a500;">
<h3 style="margin:0 0 10px 0;">🍴 {safe_display_name}</h3>
<table style="width:100%;font-size:15px;border-collapse:collapse;">
<tr><td style="padding:4px;">🔥 <b>Calories</b></td><td style="padding:4px;"><b>{calories:.1f} kcal</b></td></tr>
<tr><td style="padding:4px;">💪 <b>Protein</b></td><td style="padding:4px;">{protein:.1f} g</td></tr>
<tr><td style="padding:4px;">🌾 <b>Carbs</b></td><td style="padding:4px;">{carbs:.1f} g</td></tr>
<tr><td style="padding:4px;">🧈 <b>Fat</b></td><td style="padding:4px;">{fat:.1f} g</td></tr>
<tr><td style="padding:4px;">🌿 <b>Fiber</b></td><td style="padding:4px;">{fiber:.1f} g</td></tr>
</table>
</div>"""

        st.markdown(
            nutrition_html,
            unsafe_allow_html=True
        )

        # ------------------------------------------------------
        # NUTRITION CHART
        # ------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
        )

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
            ]
        )

        ax.set_ylabel(
            "Amount",
            color="white"
        )

        ax.set_title(
            str(display_name),
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
            "item": str(display_name),
            "calories": float(calories),
            "protein": float(protein),
            "carbs": float(carbs),
            "fat": float(fat),
            "fiber": float(fiber)
        }

    except requests.exceptions.Timeout:

        st.error(
            "❌ Nutrition service timed out. Please try again."
        )

        return None

    except requests.exceptions.RequestException as e:

        print(f"Edamam request error: {e}")

        st.error(
            "❌ Nutrition service is temporarily unavailable."
        )

        return None

    except (KeyError, IndexError, TypeError, ValueError) as e:

        print(f"Edamam response parsing error: {e}")

        st.warning(
            f"⚠️ Nutrition information couldn't be calculated "
            f"for **{serving_query}**."
        )

        return None

    except Exception as e:

        print(f"Unexpected nutrition error: {e}")

        st.error(
            "❌ Something went wrong while calculating nutrition."
        )

        return None


# ==============================================================
# MEAL SUMMARY
# ==============================================================

def show_meal_summary(nutrition_list):

    valid = [
        item
        for item in nutrition_list
        if item
    ]

    if not valid:
        return None

    total = {
        key: sum(
            item.get(key, 0)
            for item in valid
        )
        for key in (
            "calories",
            "protein",
            "carbs",
            "fat",
            "fiber"
        )
    }

    # No leading indentation -> avoids raw HTML rendering.

    summary_html = f"""<div style="background-color:rgba(0,100,0,0.75);padding:20px 24px;border-radius:12px;color:white;margin:20px 0;border-left:4px solid #00e676;">
<h2 style="margin:0 0 12px 0;">📊 Total Meal Summary</h2>
<table style="width:100%;font-size:16px;border-collapse:collapse;">
<tr><td style="padding:5px;">🔥 <b>Total Calories</b></td><td style="padding:5px;"><b>{total['calories']:.1f} kcal</b></td></tr>
<tr><td style="padding:5px;">💪 <b>Total Protein</b></td><td style="padding:5px;">{total['protein']:.1f} g</td></tr>
<tr><td style="padding:5px;">🌾 <b>Total Carbs</b></td><td style="padding:5px;">{total['carbs']:.1f} g</td></tr>
<tr><td style="padding:5px;">🧈 <b>Total Fat</b></td><td style="padding:5px;">{total['fat']:.1f} g</td></tr>
<tr><td style="padding:5px;">🌿 <b>Total Fiber</b></td><td style="padding:5px;">{total['fiber']:.1f} g</td></tr>
</table>
</div>"""

    st.markdown(
        summary_html,
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------
    # MACRO CHART
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

    if sum(sizes) > 0:

        fig, ax = plt.subplots(
            figsize=(4, 4)
        )

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        _, _, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            textprops={
                "color": "white",
                "fontsize": 10
            }
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
# PERSONALIZED / MEAL COMPLETION SUGGESTIONS
# ==============================================================

def suggest_additions(total, food_context):

    # These are general meal-completion thresholds,
    # not personalized medical targets.

    gaps = []

    if total.get("calories", 0) < 300:
        gaps.append(
            f"Energy is relatively low "
            f"({total.get('calories', 0):.0f} kcal)."
        )

    if total.get("protein", 0) < 20:
        gaps.append(
            f"Protein is relatively low "
            f"({total.get('protein', 0):.1f} g)."
        )

    if total.get("fat", 0) < 10:
        gaps.append(
            f"Healthy fat content is relatively low "
            f"({total.get('fat', 0):.1f} g)."
        )

    if total.get("carbs", 0) < 30:
        gaps.append(
            f"Carbohydrate content is relatively low "
            f"({total.get('carbs', 0):.1f} g)."
        )

    if total.get("fiber", 0) < 5:
        gaps.append(
            f"Fiber content is relatively low "
            f"({total.get('fiber', 0):.1f} g)."
        )

    if not gaps:

        st.success(
            "✅ This meal already has a good balance of the "
            "nutrients measured by the app."
        )

        return

    gap_text = "\n".join(
        f"- {gap}"
        for gap in gaps
    )

    prompt = f"""
You are a food and nutrition assistant.

The user entered or ate:

{food_context}

Current nutrition information suggests:

{gap_text}

Provide EXACTLY 3 to 5 practical foods that could complement
the existing food or meal.

RULES:

- These are general meal-completion suggestions, not medical advice.
- Do not describe the user as deficient.
- Suggestions must naturally complement the existing food.
- Respect the apparent food category.
- If the food appears vegetarian, prefer vegetarian additions.
- If it appears vegan, prefer plant-based additions.
- If the meal contains only fruit, suggest foods that pair naturally with fruit.
- For Indian foods, prefer realistic Indian meal combinations where appropriate.
- Address the nutrients listed above.
- Keep each reason to one short sentence.
- Return only JSON.
- Do not use markdown.

Return this exact structure:

[
  {
    "suggestion": "Add whole-grain toast",
    "reason": "Adds carbohydrates and fiber to complement the meal.",
    "emoji": "🍞"
  },
  {
    "suggestion": "Add a bowl of yogurt",
    "reason": "Adds protein and pairs naturally with the meal.",
    "emoji": "🥛"
  },
  {
    "suggestion": "Add a small serving of fruit",
    "reason": "Adds carbohydrates and fiber.",
    "emoji": "🍎"
  }
]
"""

    try:

        response = generate_with_fallback(
            prompt
        )

        raw_text = clean_json_response(
            response.text
        )

        suggestions = json.loads(
            raw_text
        )

        if not isinstance(suggestions, list):
            raise ValueError(
                "Suggestion response was not a list."
            )

        valid_suggestions = []

        for suggestion in suggestions[:5]:

            if not isinstance(suggestion, dict):
                continue

            title = suggestion.get(
                "suggestion"
            )

            reason = suggestion.get(
                "reason"
            )

            emoji = suggestion.get(
                "emoji",
                "🍴"
            )

            if not title or not reason:
                continue

            valid_suggestions.append(
                {
                    "suggestion": str(title),
                    "reason": str(reason),
                    "emoji": str(emoji)
                }
            )

        if not valid_suggestions:
            raise ValueError(
                "No usable suggestions returned."
            )

        heading_html = """<div style="background-color:rgba(0,0,0,0.70);padding:16px 20px;border-radius:12px;color:white;margin:20px 0 10px 0;border-left:4px solid #ffeb3b;">
<h3 style="margin:0;">💡 Suggested Additions</h3>
</div>"""

        st.markdown(
            heading_html,
            unsafe_allow_html=True
        )

        for suggestion in valid_suggestions:

            safe_emoji = html.escape(
                suggestion["emoji"]
            )

            safe_title = html.escape(
                suggestion["suggestion"]
            )

            safe_reason = html.escape(
                suggestion["reason"]
            )

            card_html = f"""<div style="margin:8px 0;padding:12px 14px;background:rgba(0,0,0,0.65);border-radius:8px;color:white;">
<span style="font-size:18px;">{safe_emoji}</span> <b>{safe_title}</b><br>
<span style="font-size:13px;color:#ddd;">{safe_reason}</span>
</div>"""

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )

    except json.JSONDecodeError as e:

        print(
            f"Gemini suggestion JSON error: {e}"
        )

        st.warning(
            "⚠️ AI suggestions couldn't be formatted correctly. "
            "Please try again."
        )

    except Exception as e:

        error_text = str(e).lower()

        print(
            f"Gemini suggestion error: {e}"
        )

        if (
            "429" in error_text
            or "quota" in error_text
            or "resource_exhausted" in error_text
            or "resource exhausted" in error_text
        ):

            st.warning(
                "⏳ AI request limit has been reached. "
                "Please wait a moment and try again."
            )

        elif (
            "503" in error_text
            or "unavailable" in error_text
            or "high demand" in error_text
            or "temporarily" in error_text
        ):

            st.warning(
                "⚠️ AI suggestions are temporarily busy. "
                "Please try again shortly."
            )

        elif (
            "api key" in error_text
            or "api_key" in error_text
            or "credentials" in error_text
        ):

            st.error(
                "❌ AI authentication failed. "
                "Please check the app configuration."
            )

        else:

            st.warning(
                "⚠️ Personalized suggestions are temporarily unavailable."
            )


# ==============================================================
# APP HEADER
# ==============================================================

st.title(
    "🍽️ MealSnap AI – Smart Nutrition Estimator"
)

st.caption(
    "Snap your food, get the nutrition!"
)


# ==============================================================
# INPUT METHOD
# ==============================================================

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
            "e.g. '2 eggs', "
            "'1 cup basmati rice', "
            "'100g paneer'"
        )
    )

    if st.button(
        "Get Nutrition Info",
        type="primary"
    ):

        food_item = food_item.strip()

        if not food_item:

            st.warning(
                "⚠️ Please enter a food item with quantity."
            )

        else:

            with st.spinner(
                "🧪 Calculating nutrition..."
            ):

                nutrition = get_nutrition(
                    food_item
                )

            if nutrition:

                with st.spinner(
                    "💡 Preparing suggestions..."
                ):

                    suggest_additions(
                        nutrition,
                        food_context=food_item
                    )


# ==============================================================
# CAMERA / IMAGE UPLOAD
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

        except Exception as e:

            print(
                f"Image loading error: {e}"
            )

            st.error(
                "❌ This image couldn't be opened. "
                "Please try another JPG, PNG or WebP image."
            )

            st.stop()

        # ------------------------------------------------------
        # DETECT FOOD
        # ------------------------------------------------------

        detected_foods = []

        with st.spinner(
            "🤖 MealSnap AI is analysing your food..."
        ):

            try:

                detected_foods = detect_food_with_gemini(
                    image
                )

            except json.JSONDecodeError as e:

                print(
                    f"Food detection JSON error: {e}"
                )

                st.error(
                    "❌ Food recognition returned an unexpected "
                    "response. Please try again."
                )

            except Exception as e:

                error_text = str(e).lower()

                print(
                    f"Food detection error: {e}"
                )

                if (
                    "429" in error_text
                    or "quota" in error_text
                    or "resource_exhausted" in error_text
                ):

                    st.error(
                        "⏳ AI request limit reached. "
                        "Please wait a moment and retry."
                    )

                elif (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "high demand" in error_text
                    or "temporarily" in error_text
                ):

                    st.error(
                        "⚠️ Food recognition is temporarily busy. "
                        "Please try again shortly."
                    )

                elif (
                    "api key" in error_text
                    or "api_key" in error_text
                    or "credentials" in error_text
                ):

                    st.error(
                        "❌ AI authentication failed."
                    )

                else:

                    st.error(
                        "❌ Food recognition couldn't be completed. "
                        "Please try again."
                    )

        # ------------------------------------------------------
        # DETECTION RESULTS
        # ------------------------------------------------------

        if not detected_foods:

            st.warning(
                """
⚠️ No food was detected.

**Try the following:**
- Make sure the food is clearly visible.
- Use good lighting.
- Avoid blurry images.
- Capture the whole plate.
- Try taking the photo from above.
"""
            )

        else:

            st.markdown(
                "### 🔍 Detected Food Items"
            )

            for food in detected_foods:

                safe_item = html.escape(
                    food["item"]
                )

                safe_serving = html.escape(
                    food["serving"]
                )

                food_html = f"""<div style="background-color:rgba(0,0,0,0.60);padding:10px 16px;border-radius:8px;color:white;margin:5px 0;border-left:3px solid #f0a500;">
🍴 <b>{safe_item}</b> &nbsp;|&nbsp; Qty: <b>{food['count']}</b> &nbsp;|&nbsp; Serving: <i>{safe_serving}</i>
</div>"""

                st.markdown(
                    food_html,
                    unsafe_allow_html=True
                )

            # --------------------------------------------------
            # NUTRITION BREAKDOWN
            # --------------------------------------------------

            st.markdown("---")

            st.markdown(
                "### 🧪 Nutrition Breakdown"
            )

            nutrition_results = []

            for food in detected_foods:

                result = get_nutrition(
                    serving_query=food["serving"],
                    display_name=(
                        f"{food['count']}× "
                        f"{food['item']}"
                    )
                )

                if result:
                    nutrition_results.append(
                        result
                    )

            # --------------------------------------------------
            # MEAL SUMMARY
            # --------------------------------------------------

            if nutrition_results:

                st.markdown("---")

                total = show_meal_summary(
                    nutrition_results
                )

                if total:

                    food_context = ", ".join(
                        f"{food['count']} {food['item']}"
                        for food in detected_foods
                    )

                    with st.spinner(
                        "💡 Preparing meal suggestions..."
                    ):

                        suggest_additions(
                            total,
                            food_context=food_context
                        )

            else:

                st.warning(
                    "⚠️ Nutrition information couldn't be calculated "
                    "for the detected foods."
                )


# ==============================================================
# DISCLAIMER
# ==============================================================

st.markdown("---")

st.caption(
    "Nutrition values and AI food recognition are estimates. "
    "Suggested additions are general information and should not "
    "be treated as personalized medical or dietary advice."
)


# ==============================================================
# CREDITS
# ==============================================================

credits_html = """<div style="text-align:center;margin-top:30px;font-size:13px;color:grey;">
© Made by <b>Pranjal Purohit</b>
</div>"""

st.markdown(
    credits_html,
    unsafe_allow_html=True
)
