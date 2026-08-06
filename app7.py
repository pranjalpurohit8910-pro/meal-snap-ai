import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from google import genai
import json
import os
from PIL import Image
import io

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
EDAMAM_APP_ID  = "e9755876"
EDAMAM_APP_KEY = "49fa98e3b702be8cf3bd51d5ffdc5a67"

# ==============================================================
# GEMINI MODEL  
# ==============================================================
@st.cache_resource
def get_gemini_model():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=800,
        ),
        system_instruction=(
            "You are a professional nutritionist and food recognition expert. "
            "Identify every food item in images with high accuracy. "
            "Output ONLY valid JSON — no markdown, no explanations, no extra text."
        ),
    )

# ==============================================================
# FOOD DETECTION  — Gemini 2.5 Flash Vision (FREE)
# ==============================================================
def detect_food_with_gemini(image: Image.Image):
    """
    Returns: [{"item": str, "count": int, "serving": str}, ...]
    """
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    model = get_gemini_model()

    prompt = """Analyze this image and identify ALL food items that are visible.

For each food item:
1. Give the most specific name possible
   (e.g. "scrambled eggs" not "eggs", "basmati rice" not "rice", "dal makhani" not "dal")
2. Count individual pieces or portions (e.g. 2 chapatis, 3 samosas, 1 bowl)
3. Estimate a realistic serving quantity for a nutrition API
   Use formats like: "2 medium apples", "1 cup cooked rice", "100g paneer", "1 slice pizza"

Return ONLY a JSON array — no markdown fences, no explanation:
[
  {"item": "food name", "count": 2, "serving": "serving string for nutrition lookup"},
  {"item": "another food", "count": 1, "serving": "serving string"}
]

Rules:
- Include ALL visible food items, even small sides and garnishes
- Count multiple pieces of the same food (e.g. 3 idlis → count 3)
- Return [] if absolutely no food is visible
- Ignore non-food objects (plates, glasses, cutlery, hands, background)"""

    response = model.generate_content([prompt, image])
    raw = response.text.strip()

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ==============================================================
# NUTRITION  — Edamam API
# ==============================================================
def get_nutrition(serving_query: str, display_name: str = None):
    if display_name is None:
        display_name = serving_query

    url = (
        f"https://api.edamam.com/api/nutrition-data"
        f"?app_id={EDAMAM_APP_ID}&app_key={EDAMAM_APP_KEY}"
        f"&ingr={requests.utils.quote(serving_query)}"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        nutrients = resp["ingredients"][0]["parsed"][0]["nutrients"]

        calories = nutrients.get("ENERC_KCAL", {}).get("quantity", 0)
        protein  = nutrients.get("PROCNT",    {}).get("quantity", 0)
        carbs    = nutrients.get("CHOCDF",    {}).get("quantity", 0)
        fat      = nutrients.get("FAT",       {}).get("quantity", 0)
        fiber    = nutrients.get("FIBTG",     {}).get("quantity", 0)

        st.markdown(
            f"""
            <div style="background-color:rgba(0,0,0,0.72);
                        padding:18px 22px; border-radius:12px;
                        color:white; margin-bottom:16px;
                        border-left:4px solid #f0a500;">
                <h3 style="margin:0 0 10px 0;">🍴 {display_name}</h3>
                <table style="width:100%; font-size:15px;">
                  <tr><td>🔥 <b>Calories</b></td><td><b>{calories:.1f} kcal</b></td></tr>
                  <tr><td>💪 <b>Protein</b></td> <td>{protein:.1f} g</td></tr>
                  <tr><td>🌾 <b>Carbs</b></td>   <td>{carbs:.1f} g</td></tr>
                  <tr><td>🧈 <b>Fat</b></td>     <td>{fat:.1f} g</td></tr>
                  <tr><td>🌿 <b>Fiber</b></td>   <td>{fiber:.1f} g</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.bar(
            ["Calories", "Protein", "Carbs", "Fat", "Fiber"],
            [calories, protein, carbs, fat, fiber],
            color=["#f0a500", "#4caf50", "#2196f3", "#f44336", "#9c27b0"],
            edgecolor="white", linewidth=0.6,
        )
        ax.set_ylabel("Amount", color="white")
        ax.set_title(display_name, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        st.pyplot(fig)
        plt.close(fig)

        return {"item": display_name, "calories": calories,
                "protein": protein, "carbs": carbs,
                "fat": fat, "fiber": fiber}

    except (KeyError, IndexError):
        st.warning(f"⚠️ Edamam couldn't find nutrition for: **{serving_query}**. Skipping.")
        return None
    except Exception as e:
        st.error(f"❌ Network error: {e}")
        return None


# ==============================================================
# MEAL SUMMARY
# ==============================================================
def show_meal_summary(nutrition_list: list):
    valid = [n for n in nutrition_list if n]
    if not valid:
        return None

    total = {k: sum(n[k] for n in valid)
             for k in ("calories", "protein", "carbs", "fat", "fiber")}

    st.markdown(
        f"""
        <div style="background-color:rgba(0,100,0,0.75);
                    padding:20px 24px; border-radius:12px;
                    color:white; margin:20px 0;
                    border-left:4px solid #00e676;">
            <h2 style="margin:0 0 12px 0;">📊 Total Meal Summary</h2>
            <table style="width:100%; font-size:16px;">
              <tr><td>🔥 <b>Total Calories</b></td><td><b>{total['calories']:.1f} kcal</b></td></tr>
              <tr><td>💪 <b>Total Protein</b></td> <td>{total['protein']:.1f} g</td></tr>
              <tr><td>🌾 <b>Total Carbs</b></td>   <td>{total['carbs']:.1f} g</td></tr>
              <tr><td>🧈 <b>Total Fat</b></td>     <td>{total['fat']:.1f} g</td></tr>
              <tr><td>🌿 <b>Total Fiber</b></td>   <td>{total['fiber']:.1f} g</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = ["Protein", "Carbs", "Fat"]
    sizes  = [total["protein"], total["carbs"], total["fat"]]
    colors = ["#4caf50", "#2196f3", "#f44336"]
    if sum(sizes) > 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        _, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "white", "fontsize": 10},
        )
        for at in autotexts:
            at.set_color("white")
        ax.set_title("Macro Distribution", color="white", fontsize=12)
        st.pyplot(fig)
        plt.close(fig)

    return total


# ==============================================================
# PERSONALIZED SUGGESTIONS  — Gemini-powered, context-aware
# ==============================================================
def suggest_additions(total: dict, food_context: str):
    """
    Calls Gemini with the actual foods eaten + nutritional gaps to produce
    suggestions that match the food category:
      - fruit meal  → suggest fruits / fruit-compatible items
      - veg meal    → suggest only vegetarian items
      - non-veg     → any protein source is fair game
      - etc.
    food_context: plain-English description of what was eaten
                  e.g. "2 apples, 1 banana" or "1 cup dal, 2 chapatis"
    """
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    # gemini-2.5-flash is a "thinking" model — internal reasoning tokens are
    # drawn from the SAME max_output_tokens budget, BEFORE the visible answer
    # is written. If reasoning eats the whole budget, response.text comes back
    # empty/blocked, which is why every call was failing. We try to disable
    # thinking outright; if the installed SDK version doesn't support that yet,
    # we fall back to a much larger token budget as a safety cushion.
    try:
        gen_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        )
    except (AttributeError, TypeError):
        gen_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048,
        )

    suggestion_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=gen_config,
    )

    # Build a clear nutritional-gap summary for the prompt
    gaps = []
    if total["calories"] < 300:
        gaps.append(f"total calories are low ({total['calories']:.0f} kcal, target ≥ 300)")
    if total["protein"] < 20:
        gaps.append(f"protein is low ({total['protein']:.1f}g, target ≥ 20g)")
    if total["fat"] < 10:
        gaps.append(f"healthy fats are low ({total['fat']:.1f}g, target ≥ 10g)")
    if total["carbs"] < 30:
        gaps.append(f"carbohydrates are low ({total['carbs']:.1f}g, target ≥ 30g)")
    if total["fiber"] < 5:
        gaps.append(f"dietary fiber is low ({total['fiber']:.1f}g, target ≥ 5g)")

    if not gaps:
        st.success("✅ Your meal looks well-balanced — no additions needed!")
        return

    gap_text = "; ".join(gaps)

    prompt = f"""A person just ate: {food_context}

Nutritional gaps in their meal: {gap_text}

Your task:
1. Identify the food category of their meal (e.g. fruits only, vegetarian Indian, non-vegetarian, vegan, mixed, etc.)
2. Suggest EXACTLY 3 to 5 specific additions (never fewer than 3, never more than 5) that:
   - Belong to the SAME food category (e.g. if they ate only fruits, suggest fruits or foods naturally eaten with fruits like yogurt, nuts, seeds — NOT eggs, bread, or meat)
   - Directly address the nutritional gaps listed above
   - Are realistic and practical to add to that meal
   - Include a SHORT reason why each helps — ONE sentence, under 15 words, to keep the response compact

Respond with ONLY the JSON array below. No reasoning, no explanation, no markdown fences, no text before or after:
[
  {{"suggestion": "Add a handful of almonds", "reason": "Rich in healthy fats, complements a fruit meal.", "emoji": "🌰"}},
  {{"suggestion": "Add a small bowl of Greek yogurt", "reason": "Pairs well with fruit and adds 10g protein.", "emoji": "🥛"}}
]"""

    def get_response_text(response):
        """Safely pull text out of a response, with a clear reason if it's missing."""
        try:
            text = response.text
            if not text or not text.strip():
                raise ValueError("Empty response text")
            return text
        except (ValueError, AttributeError, IndexError):
            candidate = response.candidates[0] if getattr(response, "candidates", None) else None
            finish_reason = getattr(candidate, "finish_reason", "UNKNOWN") if candidate else "NO_CANDIDATES"
            raise ValueError(f"no_text:{finish_reason}")

    def try_parse(raw_text):
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            parts = raw_text.split("```")
            raw_text = parts[1] if len(parts) > 1 else parts[0]
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)  # raises JSONDecodeError if malformed/truncated

        if not isinstance(parsed, list):
            raise ValueError("Response was not a JSON list")

        parsed = parsed[:5]
        if len(parsed) < 3:
            raise ValueError(f"Only received {len(parsed)} suggestion(s), need at least 3")

        for item in parsed:
            if "suggestion" not in item or "reason" not in item:
                raise KeyError("Missing required fields in a suggestion item")

        return parsed

    def render_suggestions(suggestions):
        st.markdown(
            """
            <div style="background-color:rgba(0,0,0,0.70);
                        padding:16px 20px; border-radius:12px;
                        color:white; margin-bottom:20px;
                        border-left:4px solid #ffeb3b;">
                <h3 style="margin:0 0 14px 0;">💡 Personalized Suggestions</h3>
            """,
            unsafe_allow_html=True,
        )
        for s in suggestions:
            emoji = s.get("emoji", "•")
            st.markdown(
                f"""
                <div style="margin:8px 0; padding:10px 14px;
                            background:rgba(255,255,255,0.07);
                            border-radius:8px;">
                    <span style="font-size:18px;">{emoji}</span>
                    <b> {s['suggestion']}</b><br>
                    <span style="font-size:13px; color:#ccc;">
                        {s['reason']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    last_error = None

    try:
        response = suggestion_model.generate_content(prompt)
        raw_text = get_response_text(response)
        suggestions = try_parse(raw_text)
        render_suggestions(suggestions)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        last_error = str(e)
        try:
            retry_prompt = prompt + "\n\nIMPORTANT: Keep the entire JSON response under 400 words total."
            retry_response = suggestion_model.generate_content(retry_prompt)
            retry_raw = get_response_text(retry_response)
            suggestions = try_parse(retry_raw)
            render_suggestions(suggestions)
        except Exception as e2:
            last_error = str(e2)
            st.warning("⚠️ Couldn't generate clean suggestions this time — please try again.")
            # Plain-text diagnostic only (no raw JSON/braces shown) — safe to leave in
            # while debugging; remove this expander once things look stable.
            with st.expander("Debug details"):
                st.write(f"Reason: {last_error}")

    except Exception as e:
        st.warning(f"⚠️ Could not generate personalized suggestions: {e}")


# ==============================================================
# APP LAYOUT
# ==============================================================
st.title("🍽️ MealSnap AI – Smart Nutrition Estimator")
st.caption("Snap your food, get the nutrition!")

option = st.radio("Choose Input Method:", ["🔍 Search Bar", "📸 Camera / Upload"])

# ── SEARCH BAR ─────────────────────────────────────────────────
if option == "🔍 Search Bar":
    food_item = st.text_input(
        "Enter food with quantity",
        placeholder="e.g. '2 medium apples', '1 cup basmati rice', '100g paneer'",
    )
    if st.button("Get Nutrition Info", type="primary"):
        if not food_item.strip():
            st.warning("⚠️ Please enter a food item with quantity.")
        else:
            nutrition = get_nutrition(food_item)
            if nutrition:
                # food_context for search bar = the query itself
                suggest_additions(nutrition, food_context=food_item)

# ── CAMERA / UPLOAD ────────────────────────────────────────────
elif option == "📸 Camera / Upload":

    input_mode = st.radio("Input type:", ["Camera", "Upload Image"], horizontal=True)

    uploaded_img = None
    if input_mode == "Camera":
        uploaded_img = st.camera_input("📸 Take a picture of your food")
    else:
        uploaded_img = st.file_uploader(
            "Upload a food image", type=["jpg", "jpeg", "png", "webp"]
        )

    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        st.image(image, caption="Your food image", use_container_width=False, width=400)

        if "GEMINI_API_KEY" not in st.secrets:
            st.error("❌ Gemini API key is not configured.")
            st.stop()

        with st.spinner("🤖 Meal Snap AI is analysing your food..."):
            try:
                detected_foods = detect_food_with_gemini(image)
            except json.JSONDecodeError:
                st.error("❌ Unexpected response format. Please try again.")
                detected_foods = []
            except Exception as e:
                err = str(e)
                if "API_KEY" in err or "credentials" in err.lower():
                    st.error("❌ Invalid Google API key. Please check your key.")
                elif "quota" in err.lower() or "429" in err:
                    st.error("⏳ Free-tier rate limit hit (15 req/min). Wait a moment and retry.")
                else:
                    st.error(f"❌ Detection error: {e}")
                detected_foods = []

        if not detected_foods:
            st.warning(
                "⚠️ No food detected. Tips:\n"
                "- Ensure food is clearly visible and well-lit\n"
                "- Avoid blurry or very dark images\n"
                "- Capture the whole plate from above"
            )
        else:
            st.markdown("### 🔍 Detected Food Items")
            for food in detected_foods:
                st.markdown(
                    f"""
                    <div style="background-color:rgba(0,0,0,0.60);
                                padding:10px 16px; border-radius:8px;
                                color:white; margin:5px 0;
                                border-left:3px solid #f0a500;">
                        🍴 <b>{food['item']}</b> &nbsp;|&nbsp;
                        Qty: <b>{food['count']}</b> &nbsp;|&nbsp;
                        Serving: <i>{food['serving']}</i>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("### 🧪 Nutrition Breakdown (per item)")

            nutrition_results = []
            for food in detected_foods:
                result = get_nutrition(
                    serving_query=food["serving"],
                    display_name=f"{food['count']}× {food['item']}",
                )
                nutrition_results.append(result)

            st.markdown("---")
            total = show_meal_summary(nutrition_results)
            if total:
                # Build a natural food_context string from detections
                food_context = ", ".join(
                    f"{f['count']} {f['item']}" for f in detected_foods
                )
                suggest_additions(total, food_context=food_context)


# ==============================================================
# CREDITS
# ==============================================================
st.markdown(
    """
    <div style="text-align:center; margin-top:50px; font-size:13px; color:grey;">
        © Made by <b>Pranjal Purohit</b> &nbsp;|&nbsp;
    </div>
    """,
    unsafe_allow_html=True,
)
