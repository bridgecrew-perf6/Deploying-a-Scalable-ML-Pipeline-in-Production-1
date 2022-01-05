# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, load_data
from starter.ml.model import inference, save_model, train_model, compute_model_metrics
import os
import argparse

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = load_data()


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


def main(args):
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

    slice_feature_data(model, encoder, lb,  args.feature)


def slice_feature_data(model, encoder, lb, feature):
    dataframe = pd.DataFrame(
        columns=["feature", "value", "precision", "recall", "fbeta_score"])
    for value in data[feature].unique():
        slice_data = data[data[feature] == value]
        X_test, y_test, encoder, lb = process_data(
            slice_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        y_pred = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        dataframe = dataframe.append({"feature": feature, "value": value, "precision": precision,
                                     "recall": recall, "fbeta_score": fbeta}, ignore_index=True)
    dataframe.to_csv(os.path.join("data", "slice_output.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature for slice data.')
    parser.add_argument('--feature', type=str,
                        help='Feature to slice.', default='education')
    args = parser.parse_args()

    main(args)
