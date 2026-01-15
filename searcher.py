"""
searcher.py
Gemini-powered Structured Hybrid Fashion Search
"""

import json
import chromadb
import numpy as np
from pathlib import Path
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize

from fashion_clip.fashion_clip import FashionCLIP
from rank_bm25 import BM25Okapi

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ---------------- CONFIG ----------------
class SearchConfig:
    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "fashion_db_storage"
    METADATA_PATH = BASE_DIR / "fashion_metadata.json"

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3

    TOP_K = 5
    TOP_K_CANDIDATES = 50


# ---------------- SEARCH ENGINE ----------------
class FashionSearchEngine:
    def __init__(self):
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        # Dense retrieval (FashionCLIP)
        self.fclip = FashionCLIP("fashion-clip")
        self.client = chromadb.PersistentClient(path=str(SearchConfig.DB_PATH))
        self.collection = self.client.get_collection("fashion_items")

        # Sparse retrieval (BM25 over captions)
        self._load_metadata()

        # Gemini via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro-latest",
            temperature=0.0,
            
        )

        # schema prompt
        self.prompt = PromptTemplate(
            template="""
You are a fashion query parser.

Map the query into the following predefined fields.
Return ONLY valid JSON. Do not explain.

Schema:
{{
  "colors": [],
  "upper_garment": null,
  "lower_garment": null,
  "outerwear": null,
  "accessory": null,
  "style": null,
  "environment": null
}}

Rules:
- Use null if not mentioned
- Use lowercase strings
- Do NOT invent information

Query:
"{query}"
""",
            input_variables=["query"]
        )

        self.parser = JsonOutputParser()

    # ---------------- METADATA ----------------
    def _load_metadata(self):
        with open(SearchConfig.METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.id_map = {item["id"]: item for item in self.metadata}
        corpus = [word_tokenize(item["caption"].lower()) for item in self.metadata]
        self.bm25 = BM25Okapi(corpus)

    # ---------------- LLM PARSING ----------------
    def _parse_query(self, query: str) -> Dict:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({"query": query})

    # ---------------- ATTRIBUTE MATCHING ----------------
    def _attribute_boost(self, caption: str, parsed: Dict) -> float:
        """
        Very simple, interpretable matching.
        """
        caption = caption.lower()
        score = 0.0

        for c in parsed["colors"]:
            if c in caption:
                score += 1.0

        for key in ["upper_garment", "lower_garment", "outerwear", "accessory"]:
            val = parsed.get(key)
            if val and val in caption:
                score += 1.0

        if parsed.get("style") and parsed["style"] in caption:
            score += 1.0

        if parsed.get("environment") and parsed["environment"] in caption:
            score += 1.0

        return score

    # ---------------- SEARCH ----------------
    def search(self, query: str) -> List[Dict]:
        # 1. LLM → structured intent
        parsed = self._parse_query(query)

        # 2. Dense retrieval (FashionCLIP)
        query_emb = self.fclip.encode_text([query])[0]
        dense = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=SearchConfig.TOP_K_CANDIDATES
        )

        dense_scores = {
            doc_id: 1 / (rank + 1)
            for rank, doc_id in enumerate(dense["ids"][0])
        }

        # 3. Sparse retrieval (BM25 using parsed tokens)
        tokens = (
            parsed["colors"]
            + [v for v in parsed.values() if isinstance(v, str)]
        )

        sparse_raw = self.bm25.get_scores(tokens)
        sparse_scores = {
            item["id"]: sparse_raw[i]
            for i, item in enumerate(self.metadata)
            if sparse_raw[i] > 0
        }

        # 4. Attribute boost
        attribute_scores = {
            doc_id: self._attribute_boost(item["caption"], parsed)
            for doc_id, item in self.id_map.items()
        }

        # 5. Final fusion
        final_scores = {}
        for doc_id in dense_scores:
            final_scores[doc_id] = (
                dense_scores.get(doc_id, 0) * SearchConfig.VECTOR_WEIGHT
                + sparse_scores.get(doc_id, 0) * SearchConfig.KEYWORD_WEIGHT
                + attribute_scores.get(doc_id, 0) * 0.1
            )

        # 6. Rank + return
        top_ids = sorted(final_scores, key=final_scores.get, reverse=True)[:SearchConfig.TOP_K]

        return [
            {
                "score": round(final_scores[i], 4),
                "path": self.id_map[i]["path"],
                "caption": self.id_map[i]["caption"],
                "parsed_query": parsed
            }
            for i in top_ids
        ]


# ---------------- USAGE ----------------
if __name__ == "__main__":
    engine = FashionSearchEngine()

    query = "white shirt with a red tie in a formal office"
    results = engine.search(query)

    print("\n Parsed Query:")
    print(results[0]["parsed_query"])

    print("\n Results:")
    for r in results:
        print(f"[{r['score']}] {r['caption']}")
