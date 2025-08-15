import pickle
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn estimator
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, encoder=None, lb=None, model_path="model.pkl", encoder_path="encoder.pkl", lb_path="lb.pkl"):
    """ Serializes model and encoders to files.

    Inputs
    ------
    model : sklearn estimator
        Trained machine learning model.
    encoder : OneHotEncoder, optional
        Trained encoder.
    lb : LabelBinarizer, optional
        Trained label binarizer.
    model_path, encoder_path, lb_path : str
        Paths to save the files.
    """
    joblib.dump(model, model_path)
    if encoder:
        joblib.dump(encoder, encoder_path)
    if lb:
        joblib.dump(lb, lb_path)


def load_model(model_path="model.pkl", encoder_path="encoder.pkl", lb_path="lb.pkl"):
    """ Loads model and encoders from disk. """
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model, encoder, lb


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes model metrics on a slice of the data where `column_name` == `slice_value`.
    """
    # Select the slice
    slice_df = data[data[column_name] == slice_value]

    # Process slice data
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Get predictions
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

