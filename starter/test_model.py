import numpy as np
import pytest
import pandas as pd
from starter.ml.data import load_data

from starter.ml.model import inference, compute_model_metrics, load_model
from starter.train_model import cat_features, process_data


@pytest.fixture
def data():
    return load_data()


def test_data_to_have_no_null_values(data):
    assert data.shape == data.dropna().shape


def test_range_age(data):
    assert data['age'].between(17, 90).all()


def test_slice_cat_features(data):
    model, encoder, lb = load_model()

    metrics = pd.DataFrame(
        columns=["precision", "recall", "fbeta"],
    )

    for feature in cat_features:
        for value in data[feature].unique():
            slice_data = data[data[feature] == value]
            X_test, y_test, encoder, lb = process_data(
                slice_data, categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )
            prediction = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(
                y_test, prediction)

            metrics = metrics.append({
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }, ignore_index=True)

    assert metrics.mean()['precision'] > 0.6
    assert metrics.mean()['recall'] > 0.6
    assert metrics.mean()['fbeta'] > 0.6
