"""Pluggable image-to-3D backends. Each one runs a real pretrained model or
reports why it cannot; none of them ever returns a stand-in mesh."""
from .base import Backend, GenerationRecord
from .triposg import TripoSGBackend

BACKENDS = {"triposg": TripoSGBackend}


def get_backend(name: str, **kw) -> Backend:
    try:
        return BACKENDS[name](**kw)
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}") from None
