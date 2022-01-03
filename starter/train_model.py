# Script to train machine learning model.

from re import X
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import inference, train_model, compute_model_metrics
import pickle
import os

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(os.path.join("data", "cleaned_data.csv"))


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Train and save a model.


def save_model(model, encoder, lb):
    with open(os.path.join("model", "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join("model", "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    with open(os.path.join("model", "lb.pkl"), "wb") as f:
        pickle.dump(lb, f)


def load_model():
    with open(os.path.join("model", "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join("model", "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join("model", "lb.pkl"), "rb") as f:
        lb = pickle.load(f)

    return model, encoder, lb


def main():
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)

    save_model(model, encoder, lb)

    y_pred = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {fbeta}")


if __name__ == "__main__":
    main()
