"""EgoDex human-to-human similarity retrieval; no automatic deduplication."""
__version__ = '1.0.0'

def build_index(dataset_root, annotation_root=None, config=None):
    from .features.feature_store import build_index as build
    return build(dataset_root, annotation_root, config)

from .search.retrieval_engine import RetrievalEngine
