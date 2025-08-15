import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# -----------------------------
# Load the census.csv data
# -----------------------------
project_path = os.getcwd()  # Use current working directory
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# -----------------------------
# Split the provided data into train and test sets
# -----------------------------
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# -----------------------------
# Process the data
# -----------------------------
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# -----------------------------
# Train the model
# -----------------------------
model = train_model(X_train, y_train)

# -----------------------------
# Save the model and the encoder
# -----------------------------
model_dir = os.path.join(project_path, "model")
model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")
lb_path = os.path.join(model_dir, "lb.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# -----------------------------
# Load the model (for demonstration)
# -----------------------------
model = load_model(model_path)

# -----------------------------
# Run inference on the test dataset
# -----------------------------
preds = inference(model, X_test)

# -----------------------------
# Calculate and print the metrics
# -----------------------------
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# -----------------------------
# Compute performance on model slices
# -----------------------------
with open("slice_output.txt", "w") as f:  # overwrite previous
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            precision, recall, f1 = performance_on_categorical_slice(
                test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}", file=f)

