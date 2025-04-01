from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
from sklearn.preprocessing import OrdinalEncoder
import re


with open("models/model.joblib", "rb") as file:
    model = pickle.load(file)

app = FastAPI(title="Car Price")


def get_column_types(df):
    categorical_columns = []
    numeric_columns = []

    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)

    return categorical_columns, numeric_columns


def extract_engine_features(engine_str):
    power_match = re.search(r'(\d+\.?\d*)\s*HP', engine_str)
    volume_match = re.search(r'(\d+\.?\d*)\s*L', engine_str)

    power = float(power_match.group(1)) if power_match else None

    volume = float(volume_match.group(1)) if volume_match else None

    remaining_string = engine_str
    if power_match:
        remaining_string = remaining_string.replace(power_match.group(0), '').strip()
    if volume_match:
        remaining_string = remaining_string.replace(volume_match.group(0), '').strip()

    return power, volume, remaining_string


def featurize(df):
    df[['power', 'volume', 'engine']] = df['engine'].apply(
        lambda x: pd.Series(extract_engine_features(x))
    )

    df = df.drop(columns=['milage', 'engine'], axis=1)

    return df


def transform_data(df):
    df = featurize(df)

    categorical_columns, numerical_columns = get_column_types(df)

    ordinal = OrdinalEncoder()
    ordinal.fit(df[categorical_columns])
    ordinal_encoded = ordinal.transform(df[categorical_columns])
    df_categorical = pd.DataFrame(ordinal_encoded, columns=categorical_columns)

    df[categorical_columns] = df_categorical
    if 'clean_title' in df.columns:
        df = df.drop(columns=['clean_title'], axis=1)

    return df


class CarFeatures(BaseModel):
    brand: str
    model: str
    model_year: int
    milage: int
    fuel_type: str
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    clean_title: str


@app.post("/predict", summary="Predict car price")
async def predict(car: CarFeatures):
    columns_names = [
        "brand",
        "model",
        "model_year",
        "milage",
        "fuel_type",
        "engine",
        "transmission",
        "ext_col",
        "int_col",
        "accident",
        "clean_title"
    ]

    input_data = pd.DataFrame([car.model_dump()])
    input_data.columns = columns_names
    transformed = transform_data(input_data)
    print(transformed)
    pred = model.predict(transformed)[0]

    return {"predicted_price": round(float(pred), 3)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)