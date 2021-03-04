import uvicorn

from .api import app


def main():
    """Initialize the FastAPI application"""
    uvicorn.run(app, port=5552, host="0.0.0.0")


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    main()
