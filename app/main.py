from fastapi import FastAPI, HTTPException, Header
from api_definitions import PredectionRequest, InferenceOutput
import joblib
import pandas as pd
from typing import Union, List
from numpy import ndarray

app = FastAPI()

model = joblib.load("gb_model.pkl") # Load the model
feature_names = ['x_39', 'x_49', 'x_61' , 'x_65_toyota' , 'x_75', 'x_9_monday',  'x_9_friday'] #features used in the model
relevant_features = ['x_39', 'x_49', 'x_61' , 'x_65' , 'x_75', 'x_9'] #relevant features in the request

x_39_mean = 0.252983
x_49_mean = 0.400321
x_61_mean = 0.499150
x_75_mean = 11.427923
bussiness_threshold = 0.75
   

def pre_process(full_features: List[PredectionRequest]) -> pd.DataFrame:
   
    extracted_data = [{key: getattr(item, key) for key in relevant_features} for item in full_features] # Extract relevant features
  
    df = pd.DataFrame(extracted_data)
    df.fillna({'x_39': x_39_mean, 'x_49': x_49_mean, 'x_61': x_61_mean}, inplace=True) # Fill missing values with mean

    # convert x_75 to float by removing $ and commas
    df['x_75'] = df['x_75'].str.replace('$','',regex=False)
    df['x_75'] = df['x_75'].str.replace(',','',regex=False)
    df['x_75'] = df['x_75'].str.replace(')','',regex=False)
    df['x_75'] = df['x_75'].str.replace('(','-',regex=False)
    df['x_75'] = df['x_75'].astype(float)
    df['x_75'].fillna({'x_75': x_75_mean}, inplace=True)

    # One hot encoding for categorical variables
    df['x_65_toyota'] = df['x_65'].apply(lambda x: 1 if x == 'toyota' else 0)
    df['x_9_monday'] = df['x_9'].apply(lambda x: 1 if x == 'monday' else 0)
    df['x_9_friday'] = df['x_9'].apply(lambda x: 1 if x == 'friday' else 0)
    df.drop(columns=['x_65', 'x_9'], inplace=True)
    df = df[feature_names]
    return df


def post_process(model_output: ndarray, df: pd.DataFrame) -> List[InferenceOutput]:
    last_pred = model_output[:, -1] 
    df['probability'] = last_pred
    df['business_outcome'] = df['probability'].apply(lambda x: 1 if x > bussiness_threshold else 0)

    outputs = []

    # Convert DataFrame to list of dictionaries
    records = df[feature_names + ['business_outcome', 'probability']].to_dict(orient='records')

    # Create InferenceOutput objects
    outputs = [
        InferenceOutput(
            feature_input={key: record[key] for key in feature_names},
            business_outcome=record['business_outcome'],
            probability=record['probability']
        )
        for record in records
    ]
    return outputs


@app.post("/inference/", response_model=List[InferenceOutput])
def run_inference( 
    request: Union[PredectionRequest, List[PredectionRequest]],  # Handles -d @candidate_27_test_inference.json
    accept: str = Header(None),  # Handles -H 'accept: application/json'
    content_type: str = Header(None) # Handles -H 'Content-Type: application/json'  
):
    
    if accept != "application/json" or content_type != "application/json":
        raise HTTPException(status_code=400, detail="Invalid headers")
    
    try:
        if isinstance(request, PredectionRequest):
            request = [request]
        model_input = pre_process(request)
        model_output = model.predict_proba(model_input)
        response = post_process(model_output, model_input)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Run using: uvicorn filename:app --reload