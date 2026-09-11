"""A disk-backed embedding cache for the retrieval eval.

One JSON file per (model, dimensions). Each vector is stored as base64 float32:
about 8 KB per 1536-dimension vector instead of ~30 KB of decimal text. OpenAI
returns float32-precision values, so rounding to float32 loses nothing real — and
the rounded value is also what this store hands back in the run that wrote it, so a
cold cache and a warm one rank identically.

The directory is git-ignored. Vectors are API output rather than source, and the
first run costs a fraction of a cent to regenerate.
"""

import base64
import json
import os
from collections.abc import Iterator, MutableMapping
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent / ".embedding_cache"


def _encode(vector: list[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def _decode(blob: str) -> list[float]:
    return np.frombuffer(base64.b64decode(blob), dtype="<f4").astype(float).tolist()


class JsonEmbeddingStore(MutableMapping[str, list[float]]):
    def __init__(self, model: str, dimensions: int, directory: Path = DEFAULT_DIR) -> None:
        self.model = model
        self.dimensions = dimensions
        self.path = directory / f"{model}-{dimensions}.json"
        self._vectors: dict[str, list[float]] = {}
        self.loaded = 0
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("model") != model or data.get("dimensions") != dimensions:
                raise ValueError(f"{self.path} holds {data.get('model')}/{data.get('dimensions')}")
            self._vectors = {key: _decode(blob) for key, blob in data["vectors"].items()}
            self.loaded = len(self._vectors)

    def __getitem__(self, key: str) -> list[float]:
        return self._vectors[key]

    def __setitem__(self, key: str, vector: list[float]) -> None:
        # Round on write, so this run sees exactly what the next run will load.
        self._vectors[key] = _decode(_encode(vector))

    def __delitem__(self, key: str) -> None:
        del self._vectors[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._vectors)

    def __len__(self) -> int:
        return len(self._vectors)

    def save(self) -> None:
        """Write atomically: a crash mid-write must not leave a truncated cache."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "dimensions": self.dimensions,
            "vectors": {key: _encode(vector) for key, vector in sorted(self._vectors.items())},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)
