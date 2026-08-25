"""
smell_alike  --  Drosophila-inspired similarity search
======================================================

>>> from smell_alike import FlyHash, LearnedFlyHash, TextEmbedder
"""

from smell_alike.fly_hash import FlyHash
from smell_alike.learned_fly_hash import LearnedFlyHash
from smell_alike.baselines import ClassicLSH, PCAHash
from smell_alike.embeddings import TextEmbedder, ImageEmbedder, AudioEmbedder
from smell_alike.connectome import ConnectomeFlyHash

__all__ = [
    "FlyHash", "LearnedFlyHash", "ConnectomeFlyHash",
    "ClassicLSH", "PCAHash",
    "TextEmbedder", "ImageEmbedder", "AudioEmbedder",
]
