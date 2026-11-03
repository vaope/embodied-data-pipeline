"""Frozen multilingual SentenceTransformer with token chunks to avoid silent truncation."""
import numpy as np

class SemanticEncoder:
    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(cfg['semantic_model'], revision=cfg['semantic_revision'],
                                         device=cfg['device'])
        self.model.eval()
        self.batch_size = cfg['batch_size']
        self.chunk_size = min(cfg['semantic_chunk_tokens'], self.model.max_seq_length-2)
        if self.chunk_size < 1:
            raise ValueError('invalid semantic chunk token count')
        self.metadata = {'model': cfg['semantic_model'], 'requested_revision': cfg['semantic_revision'],
                         'resolved_revision': getattr(self.model[0].auto_model.config, '_commit_hash', None),
                         'pooling': 'normalized mean of token chunks; model mean pooling per chunk',
                         'chunk_tokens': self.chunk_size}

    def encode(self, texts):
        chunks, spans = [], []
        for text in texts:
            ids = self.model.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError('empty semantic text')
            begin = len(chunks)
            chunks.extend(self.model.tokenizer.decode(ids[i:i+self.chunk_size], skip_special_tokens=True)
                          for i in range(0, len(ids), self.chunk_size))
            spans.append((begin, len(chunks)))
        encoded = self.model.encode(chunks, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        vectors = np.array([encoded[a:b].mean(axis=0) for a, b in spans], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms < 1e-12):
            raise ValueError('invalid semantic embedding')
        return vectors/norms
