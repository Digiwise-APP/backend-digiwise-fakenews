import re
import logging

from typing import Dict, List, Text, Tuple, Any
from pydantic import BaseModel, validator

from fastapi import FastAPI, HTTPException, status

import model
from scraper import Scrap

app = FastAPI()


@app.get("/")
async def main_route():
    return {"message": "Hello, World!"}


class DetectFakeNewsRequest(BaseModel):
    input_type: Text
    input_text: Text

    @validator("input_type")
    def validate_input_type(cls, value):
        if value not in ("url", "plaintext"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="input_type must either plaintext or url.",
            )

        return value


class DetectFakeNewsResponse(BaseModel):
    validity: bool
    articles: List[Dict[str, Any]]


@app.post(
    "/detect-fake-news",
    status_code=status.HTTP_200_OK,
    response_model=DetectFakeNewsResponse,
)
async def detect(*, request: DetectFakeNewsRequest) -> Any:
    text = request.input_text
    title = ""
    if request.input_type == "url":
        title, text = get_article(text)

    valid, articles = model.detect_fake_news(text, title)
    return {"validity": valid, "articles": articles}


def get_article(url: str) -> Tuple[str, str]:
    try:
        scrap = Scrap(url)
        if not scrap:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Article {url} cannot be scraped.",
            )

        title, text = scrap.title, scrap.text
        text = re.sub(r"\n", " ", text)
    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Article {url} cannot be scraped."
        )

    return title, text
