import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "Dataset"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLS = [
    "location", "city", "latitude", "longitude",
    "numBathrooms", "numBalconies", "isNegotiable", "SecurityDeposit",
    "Status", "Size_ft²", "BHK", "rooms_num",
    "property_type", "verification_days",
]
CATEGORICAL_COLS = ["location", "city", "Status", "property_type"]

def load_and_encode():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    label_encoders = {}
    
    for col in CATEGORICAL_COLS:
        train_df[col] = train_df[col].fillna("Unknown").astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("Unknown").astype(str)
            
        le = LabelEncoder()

        le.fit(train_df[col])
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        
        if col in test_df.columns:
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            test_df[col] = test_df[col].map(le_dict).fillna(-1).astype(int)
            
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")
    print("Saved models/label_encoders.pkl")
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["price"]
    
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["price"] if "price" in test_df.columns else None
    
    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"Median rent: Rs {y_train.median():,.0f}")
    
    return X_train, y_train, X_test, y_test, label_encoders

if __name__ == "__main__":
    load_and_encode()