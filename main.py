# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from starter.ml.data import process_data
from starter.ml.model import inference, load_model
import os
import pandas as pd

# Instantiate the app.
app = FastAPI()

# Heroku access to DVC data
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


class Predict(BaseModel):
    prediction: str

    @validator('*', pre=True)
    def to_string(cls, v):
        if isinstance(v, str):
            return v
        return 'Salary > 50k' if v == 1 else 'Salary <= 50k'
        
       
# Define a GET on the specified endpoint.
@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/predict/", response_model=Predict)
async def inference_data(person: Person):
    model, encoder, lb = load_model()
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

    # load predict_data
    df = pd.DataFrame(person.dict(by_alias=True), index=[0])
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb)

    prediction = inference(model, X)

    return Predict(prediction=prediction[0])
