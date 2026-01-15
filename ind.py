import json
import torch
import chromadb
import hashlib
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import traceback

# Models
from fashion_clip.fashion_clip import FashionCLIP
from transformers import BlipProcessor, BlipForConditionalGeneration


# ---------------- CONFIG ----------------
class IndexConfig:
    DATA_PATH = Path("./data")
    DB_PATH = Path("./fashion_db_storage")
    METADATA_PATH = Path("./fashion_metadata.json")

    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CAPTION_MAX_LEN = 40
    NUM_BEAMS = 3


# ---------------- INDEXER ----------------
class FashionIndexBuilder:
    def __init__(self):
        print(f"🚀 Initializing on device: {IndexConfig.DEVICE}")
        self.device = IndexConfig.DEVICE

        # FashionCLIP (image embeddings)
        self.fclip = FashionCLIP("fashion-clip")

        # BLIP (captioning)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(IndexConfig.DB_PATH)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="fashion_items"
        )

    # -------------------------------------
    def _get_images(self) -> List[Path]:
        if not IndexConfig.DATA_PATH.exists():
            raise FileNotFoundError(f"Data path not found: {IndexConfig.DATA_PATH}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = [
            p for p in IndexConfig.DATA_PATH.rglob("*")
            if p.suffix.lower() in valid_exts
        ]

        print(f"📸 Found {len(images)} images")
        return images

    # -------------------------------------
    def _generate_captions(self, images: List[Image.Image]) -> List[str]:
        inputs = self.processor(images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.caption_model.generate(
                **inputs,
                max_length=IndexConfig.CAPTION_MAX_LEN,
                num_beams=IndexConfig.NUM_BEAMS
            )

        captions = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return captions

    # -------------------------------------
    def build_index(self):
        image_paths = self._get_images()
        if not image_paths:
            print("❌ No images found. Exiting.")
            return

        metadata_export = []

        for i in tqdm(
            range(0, len(image_paths), IndexConfig.BATCH_SIZE),
            desc="Indexing"
        ):
            batch_paths = image_paths[i : i + IndexConfig.BATCH_SIZE]

            try:
                print(f"\n➡️ Processing batch {i} (size={len(batch_paths)})")

                # ---- Load images ----
                pil_images = [
                    Image.open(p).convert("RGB") for p in batch_paths
                ]

                # ---- Generate captions (BLIP) ----
                captions = self._generate_captions(pil_images)
                assert len(captions) == len(batch_paths)

                # ---- Generate embeddings (FashionCLIP) ----
                path_strings = [str(p) for p in batch_paths]
                embeddings = self.fclip.encode_images(path_strings, batch_size=len(path_strings))


                # ---- Create unique IDs ----
                ids = [
                    hashlib.md5(str(p).encode()).hexdigest()
                    for p in batch_paths
                ]

                # ---- Metadata ----
                metadatas = [
                    {"path": str(p), "caption": c}
                    for p, c in zip(batch_paths, captions)
                ]

                # ---- Store in Chroma ----
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=captions,
                    metadatas=metadatas
                )

                # ---- Export metadata ----
                for _id, meta in zip(ids, metadatas):
                    metadata_export.append({
                        "id": _id,
                        "path": meta["path"],
                        "caption": meta["caption"]
                    })

                print(f"✅ Indexed {len(batch_paths)} images")

            except Exception as e:
                print("⚠️ Batch failed")
                print(type(e).__name__, e)
                traceback.print_exc()

        # ---- Save metadata JSON ----
        with open(IndexConfig.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata_export, f, indent=2)

        print("\n🎉 Indexing complete")
        print(f"📦 Total indexed images: {len(metadata_export)}")
        print(f"🗂 Metadata saved to: {IndexConfig.METADATA_PATH}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    indexer = FashionIndexBuilder()
    indexer.build_index()
