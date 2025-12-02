import os
import joblib
import train  # модель обучится и создаст model.pkl


def test_model_file_created():
    assert os.path.exists("model.pkl"), "model.pkl was not created"


def test_model_can_predict():
    loaded = joblib.load("model.pkl")
    pred = loaded.predict([[5.1, 3.5, 1.4, 0.2]])
    assert pred is not None
