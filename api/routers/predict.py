from fastapi import APIRouter
from api.schema import Customer
from src.predictor import predict_churn
from time import perf_counter


router = APIRouter()


@router.post("/predict")
def predict(customer: Customer):

    start_time = perf_counter()

    result = predict_churn(
        customer.model_dump()
    )
    
    end_time = (perf_counter()-start_time)*1000
    result["inference_time"] = round(end_time, 2)

    return result

