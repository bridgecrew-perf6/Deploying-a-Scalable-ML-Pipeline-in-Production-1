from pickle import load
from fastapi.testclient import TestClient
from main import app
from starter.ml.data import load_data


client = TestClient(app)

# A function to test the get


def test_get():
    """
    Test the get request
    """
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():

    response = client.post("/predict/", json={
        "age": "30",
        "workclass": "Private",
        "fnlgt": "0",
        "education": "Bachelors",
        "education-num": "13",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialtys",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"})

    assert response.status_code == 200, response.json()


def test_validation_fields():

    response = client.post("/predict/", json={
        "fnlgt": "0",
        "education": "Bachelors",
        "education-num": "13",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialtys",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"})

    assert response.status_code == 422
    assert len(response.json()['detail']) == 2


def test_post_less_50k():
    """
    Test the predict output for salary >=50k.
    """
    input_dict = {
        "age": 20,
        "workclass": "State-gov",
        "fnlgt": 0,
        "education": "Bachelors",
        "education-num": 8,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict/", json=input_dict)
    assert response.status_code == 200
    assert response.json()["prediction"] == "Salary <= 50k"


def test_post_greater_50k():
    input_dict = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 1020,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    response = client.post("/predict/", json=input_dict)
    assert response.status_code == 200
    assert response.json()["prediction"] == "Salary > 50k"
