import requests

data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 15781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1020,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
}

response = requests.post(
    url='https://fastapi-pipeline-ml.herokuapp.com/predict/',
    json=data
)

print(response.json())