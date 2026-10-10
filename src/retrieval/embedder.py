"""
Batch-encode incident texts to embeddings using sentence-transformers.
"""
import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class IncidentEmbedder:
    """Encodes incident text descriptions to embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedder with a sentence-transformer model.

        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Model loaded. Embedding dimension: {self.get_embedding_dimension()}")

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """
        Encode a list of texts to embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding (default 128).

        Returns:
            Array of embeddings (shape: [len(texts), embedding_dim]).
        """
        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        logger.info(f"Encoded shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2).
        """
        # sentence-transformers 6 renamed the method; keep older versions working.
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return getter()
