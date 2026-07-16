from fastapi import FastAPI

from api.routers import predict
from api.routers import health



app = FastAPI(
    title="Churn Prediction API"
)


app.include_router(
    predict.router
)

app.include_router(
    health.router
)