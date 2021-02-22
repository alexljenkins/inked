import base64
import io
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from ..character import FixedSpacer
from ..generator import WordGenerator

logger = logging.getLogger()

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
factory = WordGenerator(augmentor=True, warehouses=["lmdb", "fonts"])


def pil_2_base64(image: Image.Image) -> str:
    """Converts PIL Image into base64 encoded string for use in HTML"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{img_str}"


@router.get("/", description="Demo UI for the post router", response_class=HTMLResponse)
async def typesetter(request: Request):
    """Demo UI for the post router
    """
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/", description="Produces an image for the given text and configuration")
async def typesetter_generator(word: str):
    """Produces an image for the given text and configuration
    """
    word_gen = factory.generate(word, augment_word=True, spacer=FixedSpacer(5))
    return {"img": pil_2_base64(word_gen.image)}
