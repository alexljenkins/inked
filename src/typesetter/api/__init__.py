from fastapi import FastAPI

from .typesetter_router import router as typesetter

app = FastAPI(
    version="0.1.0",
    title="Typesetter",
    description="Microservice that uses typesetter to produce an image from text",
    docs_url="/",
)

responses = {
    200: {"description": "Request was successful"},
    400: {"description": "Request was unsuccessful. Client error"},
    500: {"description": "Request was unsuccessful. Server error"},
}

app.include_router(typesetter, prefix="/typesetter", tags=["typesetter"], responses=responses)  # type: ignore
