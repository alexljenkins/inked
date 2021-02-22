import os
import uvicorn

from .api import app


def main():
    """Initialize the FastAPI application"""
    uvicorn.run(app, port=int(os.environ.get("PORT", 5000)), host="0.0.0.0")


if __name__ == "__main__":
    main()
