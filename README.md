# model_api_deployment

This project creates an inference API for model that was trained using the GradientBoosting Algorithm from scikit learn using FastAPI. 

The API takes 100 features as input but only 6 of the features are needed to create the the final 7 processed features for inference. It supports both single and batch requests

Docker is used to containerize the application and the model returns a json output of the business outcome, class probability, and final input features used to make a decision

## Run this app 

```bash
run_api.sh

