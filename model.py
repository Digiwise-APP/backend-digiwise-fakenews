from typing import Optional
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scraper import Scrap

model_checkpoint = "Rifky/indobert-hoax-classification"
base_model_checkpoint = "indobenchmark/indobert-base-p1"
data_checkpoint = "Rifky/indonesian-hoax-news"
label = {0: "valid", 1: "fake"}


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )
    base_model = SentenceTransformer(base_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, fast=True)
    data = load_dataset(
        data_checkpoint, split="train", download_mode="reuse_cache_if_exists"
    )
    return model, base_model, tokenizer, data


model, base_model, tokenizer, data = load_model()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def detect_fake_news(text: str, title: Optional[str]):
    token = text.split()
    text_len = len(token)

    sequences = []
    for i in range(text_len // 512):
        sequences.append(" ".join(token[i * 512 : (i + 1) * 512]))
    sequences.append(" ".join(token[text_len - (text_len % 512) : text_len]))
    sequences = tokenizer(
        sequences,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    predictions = model(**sequences)[0].detach().numpy()
    result = [
        np.sum([sigmoid(i[0]) for i in predictions]) / len(predictions),
        np.sum([sigmoid(i[1]) for i in predictions]) / len(predictions),
    ]

    title_embeddings = base_model.encode(title)
    similarity_score = cosine_similarity(
        [title_embeddings], data["embeddings"]
    ).flatten()
    sorted = np.argsort(similarity_score)[::-1].tolist()

    prediction = np.argmax(result, axis=-1)

    related_articles = []
    for i in sorted[:5]:
        article = {
            "domain_url": data["url"][i].split("/")[2],
            "url": data["url"][i],
            "title": data["title"][i],
        }
        related_articles.append(article)

    valid = True if prediction == 0 else False

    return valid, related_articles
