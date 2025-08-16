import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


# Load a small sample of the census data for testing
@pytest.fixture(scope="module")
def sample_data():
    data = pd.read_csv("data/census.csv")
    data = data.sample(500, random_state=42)  # small subset for speed
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    X, y, encoder, lb = process_data(
        data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    return X, y, encoder, lb


def test_train_model_returns_randomforest(sample_data):
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    X, y, _, _ = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_output_shape(sample_data):
    """
    Test that inference returns predictions of the expected shape.
    """
    X, y, _, _ = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_values(sample_data):
    """
    Test that compute_model_metrics returns non-negative values.
    """
    X, y, _, _ = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0

