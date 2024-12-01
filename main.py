import asyncio
from typing import Optional
from sklearn.pipeline import Pipeline

import os
import time
import mlflow
import base64
import logging
import numpy as np
import pandas as pd
import streamlit as st


# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup page.
about = """This is a mobile phone price range classifier. This app is build in association with *Unified Mentor* for machine learning project submition. The classifier models were trained on dataset provided by *Unified Mentor*. I'm thankful to *Unified Mentor* to provide this platform."""

st.set_page_config(page_title="Mobile Phone Price Classifier",
                   page_icon="📱", menu_items={"About": f"{about}"})
st.title(body="Is Your Dream Phone Too Expensive? Check Now with 'Phone Price Range Classifier!'📱💸")
st.markdown(
    "*Estimate the price range of your desired phone using machine learning models.*")

# Initialize session state
if 'sk_model_log_reg' not in st.session_state:
    st.session_state.sk_model_log_reg = None
if 'sk_model_knn' not in st.session_state:
    st.session_state.sk_model_knn = None
if 'sk_model_random_forest' not in st.session_state:
    st.session_state.sk_model_random_forest = None

# Model loading function


@st.cache_resource
def load_model(model_name: str) -> Optional[Pipeline]:
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        model_uri = f"models:/{model_name}@champion"
        sk_model = mlflow.sklearn.load_model(model_uri=model_uri)
        # sk_model = await asyncio.to_thread(mlflow.sklearn.load_model, model_uri=model_uri)
        # st.success(f"{model_name} model loaded successfully!", icon="✅")
        # await asyncio.sleep(2)  # Simulate loading time, remove in production
        return sk_model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f":red[Failed to load *{model_name}* model. Please try again later.]", icon="🚨")
        return None


# Load models
# async def load_models() -> None:
if 'sk_model_log_reg' not in st.session_state or st.session_state.sk_model_log_reg is None:
    with st.spinner(f":green[This may take a few moments. Loading *Logistic Regression* model...]"):
        st.session_state.sk_model_log_reg = load_model("mob_price_log_reg")
if 'sk_model_knn' not in st.session_state or st.session_state.sk_model_knn is None:
    with st.spinner(f":green[This may take a few moments. Loading *KNN* model...]"):
        st.session_state.sk_model_knn = load_model("mob_price_knn")
if 'sk_model_random_forest' not in st.session_state or st.session_state.sk_model_random_forest is None:
    with st.spinner(f":green[This may take a few moments. Loading *Random Forest Classifier* model...]"):
        st.session_state.sk_model_random_forest = load_model(
    "mob_price_random_forest")

# asyncio.run(load_models())

# Feature selecting column
st.markdown("""----""")
st.header("Choose Your Desired Mobile Features Below")

# columns = ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"]

b_power = st.slider("Battery Power (mAh) [*20.07% in correlation with price range*]", min_value=500.0,
                    max_value=2000.0, value=1200.0)
talk_time = st.slider("Battery Backup (hrs) [*2.19% in correlation with price range*]",
                      min_value=2.0, max_value=20.0, value=8.0)
ram = st.slider("RAM (MB) [*91.70% in correlation with price range*]",
                min_value=200.0, max_value=4000.0, value=2000.0)
int_mem = st.slider("Internal Memory (GB) [*4.44% in correlation with price range*]", min_value=2.0,
                    max_value=64.0, value=4.0)
clk_speed = st.slider("Processor Speed [*0.66% in correlation with price range*]", min_value=0.5,
                      max_value=3.0, value=2.0)
n_cores = st.slider(
    "Number of Cores [*0.44% in correlation with price range*]", min_value=1.0, max_value=8.0, value=8.0)
pri_camera = st.slider("Primary Camera (MP) [*3.36% in correlation with price range*]",
                       min_value=0.0, max_value=20.0, value=12.0)
front_camera = st.slider(
    "Front Camera (MP) [*2.2% in correlation with price range*]", min_value=0.0, max_value=19.0, value=5.0)
px_height = st.slider("Pixel Resolution Height (pixels) [*14.89% in correlation with price range*]",
                      min_value=0.0, max_value=2000.0, value=2000.0)
px_width = st.slider("Pixel Resolution Widht (pixels) [*16.58% in correlation with price range*]",
                     min_value=500.0, max_value=2000.0, value=1080.0)
sc_height = st.slider("Screen Height (cm) [*2.3% in correlation with price range*]", min_value=5.0,
                      max_value=19.0, value=15.3)
sc_width = st.slider("Screen Width (cm) [*3.87% in correlation with price range*]", min_value=0.0,
                     max_value=18.0, value=7.66)
mob_depth = st.slider("Mobile Depth (cm) [*0.09% in correlation with price range*]", min_value=0.1,
                      max_value=1.0, value=0.98)
mobile_wt = st.slider("Mobile Weight (g) [*-3.03% in correlation with price range*]", min_value=80.0,
                      max_value=200.0, value=155.0)
blue = st.radio("Bluetooth [*2.06% in correlation with price range*]", [
                "**1** :thumbsup:", "**0** :thumbsdown:"], captions=["Available", "Unavailable"])
dual_sim = st.radio("Dual SIM [*1.74% in correlation with price range*]", [
                    "**1** :thumbsup:", "**0** :thumbsdown:"], captions=["Available", "Unavailable"])
four_g = st.radio("4G [*1.48% in correlation with price range*]", ["**1** :thumbsup:", "**0** :thumbsdown:"],
                  captions=["Available", "Unavailable"])
three_g = st.radio("3G [*2.36% in correlation with price range*]", ["**1** :thumbsup:", "**0** :thumbsdown:"],
                   captions=["Available", "Unavailable"])
wifi = st.radio("WiFi [*1.88% in correlation with price range*]", ["**1** :thumbsup:", "**0** :thumbsdown:"],
                captions=["Available", "Unavailable"])
touch_screen = st.radio("Touch Screen [*-3.04% in correlation with price range*]",
                        ["**1** :thumbsup:", "**0** :thumbsdown:"], captions=["Available", "Unavailable"])

# Models options
option_map = {
    "Logistic Regression": st.session_state.sk_model_log_reg,
    "KNN": st.session_state.sk_model_knn,
    "Random Forest Classifier": st.session_state.sk_model_random_forest
}
sk_model = st.pills("Select your preferred model (or leave to default) to get price range of your desired phone", options=option_map.keys(),
                    default="Logistic Regression",
                    help="""Model's Test accuracy:  *Logistic Regression 84%*,  *KNN 57.5%,  Random Forest Classifier 91%*""",
                    selection_mode="single"
                    )
# st.write(sk_model)
match sk_model:
    case "Logistic Regression":
        model = st.session_state.sk_model_log_reg
    case "KNN":
        model = st.session_state.sk_model_knn
    case "Random Forest Classifier":
        model = st.session_state.sk_model_random_forest
# st.write(model)
predict = st.button('Predict')

# Create a dataframe to store the selected features
df = pd.DataFrame(
    data={
        "battery_power": np.array([float(b_power)]), "blue": np.array([float(blue[2])]), "clock_speed": np.array([float(clk_speed)]), "dual_sim": np.array([float(dual_sim[2])]), "fc": np.array([float(front_camera)]), "four_g": np.array([float(four_g[2])]), "int_memory": np.array([float(int_mem)]), "m_dep": np.array([float(mob_depth)]), "mobile_wt": np.array([float(mobile_wt)]), "n_cores": np.array([float(n_cores)]), "pc": np.array([float(pri_camera)]), "px_height": np.array([float(px_height)]), "px_width": np.array([float(px_width)]), "ram": np.array([float(ram)]), "sc_h": np.array([float(sc_height)]), "sc_w": np.array([float(sc_width)]), "talk_time": np.array([float(talk_time)]), "three_g": np.array([float(three_g[2])]), "touch_screen": np.array([float(touch_screen[2])]), "wifi": np.array([float(wifi[2])])
    }
)

# Price range
price_range = {0: "Low Cost", 1: "Medium Cost",
               2: "High Cost", 3: "Very High Cost"}


def predict_price(model: Pipeline, df: pd.DataFrame = df) -> tuple:
    """Predict the price of a mobile phone based on the selected features."""
    preds = model.predict(df)
    pred_probs = model.predict_proba(df)
    return preds, pred_probs

# Refresh models
if st.button("Refresh Models"):
    st.cache_resource.clear()  # Clear the cache
    with st.spinner(f":green[This may take a few moments. Refreshing *Logistic Regression* model...]"):
        st.session_state.sk_model_log_reg = load_model("mob_price_log_reg")
    with st.spinner(f":green[This may take a few moments. Refreshing *KNN* model...]"):
        st.session_state.sk_model_knn = load_model("mob_price_knn")
    with st.spinner(f":green[This may take a few moments. Refreshing *Random Forest Classifier* model...]"):
        st.session_state.sk_model_random_forest = load_model("mob_price_random_forest")
    # st.markdown("""----""")
    st.write("Models reloaded successfully!")

# Run the prediction on button press
if predict:
    preds, probs = predict_price(model, df)

    # Write prediction to the page
    st.markdown("""----""")
    st.header("Price Range Prediction")
    txt = f"Your desired mobile phone will fall under following price range: ***{
        price_range[preds[0]]}***"
    st.subheader(txt)
    if np.max(probs) > 0.79:
        txt2 = f":green[{np.max(probs):.2%}]"
    elif np.max(probs) > 0.49:
        txt2 = f":blue[{np.max(probs):.2%}]"
    else:
        txt2 = f":red[{np.max(probs):.2%}]"
    st.write(f"Model's Confidence on Prediction: {txt2}")

# Disclamer
st.write("\n"*3)
st.markdown("""----""")
st.write("""*Disclamer: Predictions made by the models may be inaccurate due to the nature of the models. This is a simple demonstration of how machine learning can be used to make predictions. For more accurate predictions, consider using more complex models and larger datasets. Also, after getting your predictions from above models visit/consult with your nearest mobile store for more accurate price information for your desired mobile phone specifications.*""")


def get_image_base64(image_path):
    """Read image file."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
st.markdown("""---""")
st.markdown("Created by [Pranay Jagtap](https://pranayjagtap.netlify.app)")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay_sq.jpg")

# Create the HTML for the circular image
html_code = f"""
<style>
    .circular-image {{
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }}
    .circular-image img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>
<div class="circular-image">
    <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
</div>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
# st.image("assets/pranay_sq.jpg", width=125)
st.markdown("Electrical Engineer | Machine Learning Enthusiast"\
            "<br>📍 Nagpur, Maharashtra, India", unsafe_allow_html=True)
