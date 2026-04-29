import joblib

def load_model():
    return joblib.load("models/xgb_model.pkl")

def predict(model, X):
    preds = model.predict(X)

    # ✅ Ensure realistic output
    preds = preds.clip(min=0)

    return preds