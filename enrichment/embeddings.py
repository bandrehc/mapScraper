"""Sentence embedding generation with disk caching.

Uses sentence-transformers (all-MiniLM-L6-v2 by default).
The first call downloads ~90 MB of model weights; subsequent calls load from
HuggingFace's local cache.  Embedding arrays are cached to disk keyed by an
MD5 hash of the input texts so re-running the same dataset costs nothing.
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'all-MiniLM-L6-v2'
_CACHE_DIR = Path('data/embeddings_cache')


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _texts_hash(texts: List[str], model: str) -> str:
    content = (model + '\n' + '\n'.join(texts)).encode('utf-8')
    return hashlib.md5(content).hexdigest()[:16]


def _cache_path(texts: List[str], model: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = model.replace('/', '_')
    return _CACHE_DIR / f'{tag}_{_texts_hash(texts, model)}.npy'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: List[str],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    use_cache: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Return a float32 ndarray of shape (n, embedding_dim).

    Parameters
    ----------
    texts:          One string per lead (from text_builder.build_texts).
    model_name:     Any sentence-transformers model name or local path.
    batch_size:     Encoding batch size (reduce if OOM).
    use_cache:      Load/save embeddings from disk to avoid recomputation.
    show_progress:  Show tqdm bar during encoding.
    """
    if use_cache:
        path = _cache_path(texts, model_name)
        if path.exists():
            logger.info("Loading cached embeddings from %s", path)
            return np.load(path)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding generation.\n"
            "Install it with:  pip install sentence-transformers"
        )

    logger.info("Loading model '%s' …", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Encoding %d texts (batch_size=%d) …", len(texts), batch_size)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise → cosine similarity = dot product
    )

    if use_cache:
        path = _cache_path(texts, model_name)
        np.save(path, embeddings)
        logger.info("Embeddings cached to %s", path)

    return embeddings


def load_or_generate(
    texts: List[str],
    cache_file: Optional[str] = None,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Convenience wrapper: load from an explicit path if supplied, else generate."""
    if cache_file:
        p = Path(cache_file)
        if p.exists():
            logger.info("Loading embeddings from %s", p)
            return np.load(p)

    emb = generate_embeddings(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        use_cache=True,
        show_progress=show_progress,
    )

    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, emb)
        logger.info("Embeddings saved to %s", cache_file)

    return emb
