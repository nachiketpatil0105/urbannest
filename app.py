import sys
import joblib
import subprocess
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="UrbanNest Rent Predictor", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_rf_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoders.pkl"
TRAIN_PATH = BASE_DIR / "Dataset" / "train.csv"
TEST_PATH = BASE_DIR / "Dataset" / "test.csv"

def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.readline().strip() == "version https://git-lfs.github.com/spec/v1"
    except (UnicodeDecodeError, OSError):
        return False

def _run_training():
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "main.py")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    needs_training = (
        not MODEL_PATH.exists()
        or not ENCODER_PATH.exists()
        or _is_lfs_pointer(MODEL_PATH)
        or _is_lfs_pointer(ENCODER_PATH)
    )
    if needs_training:
        with st.spinner("Model not found. Running training pipeline, this takes a few minutes..."):
            _run_training()
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

@st.cache_data(show_spinner=False)
def load_mappings():
    both = pd.concat([pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)], ignore_index=True)
    
    both["latitude"] = pd.to_numeric(both["latitude"], errors="coerce")
    both["longitude"] = pd.to_numeric(both["longitude"], errors="coerce")
    
    city_locations = {
        city: sorted(both[both["city"] == city]["location"].dropna().unique().tolist())
        for city in sorted(both["city"].dropna().unique())
    }
    loc_coords = {
        loc: (round(float(r["latitude"]), 5), round(float(r["longitude"]), 5))
        for loc, r in both.groupby("location")[["latitude", "longitude"]].median().iterrows()
    }
    city_coords = {
        city: (round(float(r["latitude"]), 5), round(float(r["longitude"]), 5))
        for city, r in both.groupby("city")[["latitude", "longitude"]].median().iterrows()
    }
    return city_locations, loc_coords, city_coords

def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return -1  

try:
    model, label_encoders = load_artifacts()
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

city_locations, loc_coords, city_coords = load_mappings()

st.title("UrbanNest Rent Predictor")
st.caption("Rental price estimates for Mumbai, Pune, Delhi, and Hisar based on 11,000+ listings.")
st.divider()

st.subheader("Location")
col1, col2 = st.columns(2)

city = col1.selectbox("City", options=sorted(city_locations.keys()), index=list(sorted(city_locations.keys())).index("Mumbai"))
location = col2.selectbox("Locality", options=city_locations[city])

lat, lon = loc_coords.get(location, city_coords.get(city, (19.0, 73.0)))

with st.expander("Resolved Coordinates"):
    cc1, cc2 = st.columns(2)
    cc1.metric("Latitude", f"{lat:.5f}")
    cc2.metric("Longitude", f"{lon:.5f}")
    
st.divider()

with st.form("prediction_form", border=False):
    st.subheader("Property Details")
    r1, r2, r3 = st.columns(3)
    prop_type = r1.selectbox("Property Type", options=label_encoders["property_type"].classes_.tolist())
    status = r2.selectbox("Furnishing Status", options=label_encoders["Status"].classes_.tolist())
    bhk = r3.selectbox("BHK / RK", options=[("RK (Studio)", 0), ("BHK", 1)], format_func=lambda x: x[0], index=1)[1]
    
    size = r1.number_input("Size (sq ft)", min_value=100, max_value=15000, value=900, step=50)
    rooms = r2.number_input("Total Rooms", min_value=1, max_value=12, value=2)
    bathrooms = r3.number_input("Bathrooms", min_value=0, max_value=10, value=1)
    
    st.divider()
    st.subheader("Financial and Other Details")
    f1, f2, f3 = st.columns(3)
    deposit = f1.number_input("Security Deposit (Rs)", min_value=0, max_value=12_000_000, value=0, step=5_000)
    balconies = f2.number_input("Balconies", min_value=0, max_value=8, value=0)
    verification_days = f3.number_input("Listing Age (days)", min_value=0, max_value=1825, value=30)
    negotiable = f1.checkbox("Price is Negotiable")
    
    st.divider()
    submitted = st.form_submit_button("Predict Monthly Rent", type="primary", width="stretch")

if submitted:
    input_df = pd.DataFrame([{
        "location": safe_encode(label_encoders["location"], location),
        "city": safe_encode(label_encoders["city"], city),
        "latitude": lat,
        "longitude": lon,
        "numBathrooms": bathrooms,
        "numBalconies": balconies,
        "isNegotiable": int(negotiable),
        "SecurityDeposit": deposit,
        "Status": safe_encode(label_encoders["Status"], status),
        "Size_ft²": size,
        "BHK": bhk,
        "rooms_num": rooms,
        "property_type": safe_encode(label_encoders["property_type"], prop_type),
        "verification_days": verification_days,
    }])
    
    prediction = int(model.predict(input_df)[0])
    
    st.success(f"Estimated Monthly Rent: Rs {prediction:,}")
    with st.expander("Prediction Input Summary"):
        summary = pd.DataFrame({
            "Field": ["City", "Location", "Coordinates", "Property Type", "Status", "BHK",
                      "Size (sq ft)", "Rooms", "Bathrooms", "Balconies", "Security Deposit",
                      "Listing Age (days)", "Negotiable"],
            "Value": [city, location, f"({lat:.5f}, {lon:.5f})", prop_type, status,
                      "RK" if bhk == 0 else "BHK", f"{size:,}", rooms, bathrooms, balconies,
                      f"Rs {deposit:,}", verification_days, "Yes" if negotiable else "No"],
        })
        
        summary["Value"] = summary["Value"].astype(str)
        
        st.dataframe(summary, width="stretch", hide_index=True)