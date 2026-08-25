"""
embeddings.py  --  Phase 3: Domain front-ends
==============================================
Turns real-world content into L2-normalised vectors suitable for FlyHash.

Supported domains:
  - text / notes  (sentence-transformers  model: all-MiniLM-L6-v2, 384-D)
  - images        (CLIP ViT-B/32 via sentence-transformers, 512-D)

Usage:
    from smell_alike.embeddings import TextEmbedder, ImageEmbedder
    emb = TextEmbedder()
    vecs = emb.encode(["note one", "note two"])
"""
import numpy as np

# ── Text / Notes ──────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    Encode text strings into L2-normalised 384-D sentence embeddings.
    Uses all-MiniLM-L6-v2 (fast, good quality, 80 MB).

    Parameters
    ----------
    model_name : str  -- HuggingFace model identifier
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not found. "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        self._model = None   # lazy load on first use

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts, batch_size=64, show_progress=False):
        """
        Encode a list of strings into an (N, 384) float32 array.
        Vectors are L2-normalised.

        Parameters
        ----------
        texts        : list[str]
        batch_size   : int
        show_progress: bool

        Returns
        -------
        embeddings : np.ndarray, shape (N, 384), float32, L2-normalised
        """
        self._load()
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalise
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    @property
    def dim(self):
        self._load()
        getter = getattr(self._model, "get_embedding_dimension",
                         self._model.get_sentence_embedding_dimension)
        return getter()

    def __repr__(self):
        return f"TextEmbedder(model={self.model_name!r})"


# ── Images ────────────────────────────────────────────────────────────────

class ImageEmbedder:
    """
    Encode images into L2-normalised CLIP embeddings (512-D, ViT-B/32).

    Uses sentence-transformers' built-in CLIP support — no extra dependency
    beyond sentence-transformers + torch (already installed).

    Parameters
    ----------
    model_name : str  -- sentence-transformers CLIP model identifier
    """

    def __init__(self, model_name="clip-ViT-B-32"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not found. "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        self._model = None   # lazy load on first use

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, images, batch_size=32, show_progress=False):
        """
        Encode images into an (N, 512) float32 array, L2-normalised.

        Parameters
        ----------
        images        : list of PIL.Image.Image, file paths, URLs, or
                        numpy arrays (H, W, 3) uint8
        batch_size    : int
        show_progress : bool

        Returns
        -------
        embeddings : np.ndarray, shape (N, 512), float32, L2-normalised
        """
        from PIL import Image as PILImage

        # Normalise mixed inputs: numpy arrays -> PIL objects
        prepared = []
        for img in images:
            if isinstance(img, np.ndarray):
                prepared.append(PILImage.fromarray(img))
            else:
                prepared.append(img)  # PIL Image, path, or URL passthrough

        self._load()
        vecs = self._model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalise
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    @property
    def dim(self):
        self._load()
        getter = getattr(self._model, "get_embedding_dimension",
                         self._model.get_sentence_embedding_dimension)
        return getter()

    def __repr__(self):
        return f"ImageEmbedder(model={self.model_name!r})"


# ── Image corpus helpers ──────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_folder(path, extensions=None):
    """
    Collect image file paths from a folder (non-recursive), sorted by name.

    Parameters
    ----------
    path       : str  -- folder containing images
    extensions : set[str] -- allowed extensions (default: common raster formats)

    Returns
    -------
    paths : list[str]
    """
    from pathlib import Path

    exts = extensions or _IMAGE_EXTS
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    paths = sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return paths


def load_synthetic_image_corpus(n=40, size=64, seed=0):
    """
    Built-in corpus of n synthetic PIL images (no download needed) —
    stripes and solid blocks in random palettes. Groups of similar-looking
    images let you sanity-check image similarity search offline.

    Returns
    -------
    (images, labels) : tuple[list[PIL.Image.Image], list[str]]
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    palettes = [
        [(228, 26, 28), (255, 255, 204)],   # red / cream
        [(55, 126, 184), (228, 245, 250)],  # blue / ice
        [(77, 175, 74), (240, 255, 240)],   # green / mist
    ]
    kinds = ["h-stripes", "v-stripes", "diag", "block"]
    images, labels = [], []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        pal = palettes[int(rng.integers(0, len(palettes)))]
        bg, fg = pal[1], pal[0]
        img = Image.new("RGB", (size, size), bg)
        d = ImageDraw.Draw(img)
        if kind == "h-stripes":
            for y in range(0, size, 8):
                d.rectangle([0, y, size, y + 3], fill=fg)
        elif kind == "v-stripes":
            for x in range(0, size, 8):
                d.rectangle([x, 0, x + 3, size], fill=fg)
        elif kind == "diag":
            for o in range(-size, 2 * size, 12):
                d.line([(o, 0), (o + size, size)], fill=fg, width=3)
        else:  # block
            m = size // 5
            d.rectangle([m, m, size - m, size - m], fill=fg)
        images.append(img)
        labels.append(f"{kind}-{i:03d}")
    return images, labels


# ── Corpus helpers ────────────────────────────────────────────────────────────

def load_notes_corpus(path):
    """
    Load a plain text file where each non-empty line is one note/sentence.

    Returns
    -------
    texts : list[str]
    """
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def load_sample_corpus():
    """
    Returns a built-in sample corpus of 60 short knowledge snippets
    for quick demos without needing an external file.
    """
    return [
        # Biology
        "The mitochondria is the powerhouse of the cell.",
        "DNA is a double helix made of nucleotide base pairs.",
        "Neurons transmit signals via electrochemical impulses.",
        "Photosynthesis converts sunlight into glucose in plant cells.",
        "The human genome contains roughly 3 billion base pairs.",
        "CRISPR-Cas9 is a gene editing tool derived from bacteria.",
        "Enzymes are biological catalysts that speed up chemical reactions.",
        "The immune system fights pathogens using T-cells and antibodies.",
        "Epigenetics studies heritable changes not involving DNA sequence.",
        "Stem cells can differentiate into many specialised cell types.",
        # Neuroscience
        "The fly olfactory circuit implements locality-sensitive hashing.",
        "Kenyon cells in the mushroom body encode odour memories.",
        "Long-term potentiation strengthens synaptic connections.",
        "The hippocampus plays a key role in spatial and episodic memory.",
        "Dopamine is a neurotransmitter linked to reward and motivation.",
        "The prefrontal cortex regulates decision-making and planning.",
        "Mirror neurons fire both when acting and observing an action.",
        "Sleep consolidates memories by replaying neural activity.",
        "The cerebellum coordinates voluntary movement and balance.",
        "Neuroplasticity allows the brain to reorganise after injury.",
        # Computer Science
        "Locality-sensitive hashing finds approximate nearest neighbours fast.",
        "A hash function maps data to fixed-size buckets.",
        "Random projection preserves pairwise distances in expectation.",
        "Cosine similarity measures the angle between two vectors.",
        "K-nearest-neighbour search scales poorly with dimension (curse).",
        "Transformers use self-attention to model sequence dependencies.",
        "Embeddings map discrete tokens to continuous vector spaces.",
        "Dimensionality reduction compresses data while preserving structure.",
        "Binary codes allow fast Hamming-distance comparisons.",
        "Approximate search trades recall for speed and memory.",
        # Physics
        "General relativity describes gravity as spacetime curvature.",
        "Quantum entanglement links particles across any distance.",
        "The Higgs boson gives particles mass via the Higgs field.",
        "Thermodynamics governs energy transfer and entropy.",
        "Wave-particle duality means photons behave as both.",
        "Black holes have a singularity inside an event horizon.",
        "Dark matter makes up about 27% of the universe.",
        "Special relativity says time dilates at high speeds.",
        "Superconductors conduct electricity with zero resistance.",
        "Quantum computers use qubits instead of classical bits.",
        # Math
        "A prime number has no divisors other than 1 and itself.",
        "The Fourier transform decomposes a signal into frequencies.",
        "Gradient descent minimises a loss function iteratively.",
        "Bayesian inference updates beliefs given new evidence.",
        "The central limit theorem explains why normal distributions appear.",
        "Linear algebra underlies most modern machine learning.",
        "Topology studies properties preserved under continuous deformation.",
        "Eigenvalues describe the scaling factor of eigenvectors.",
        "Monte Carlo methods estimate integrals via random sampling.",
        "Graph theory studies nodes connected by edges.",
        # History / Philosophy
        "The printing press accelerated the spread of knowledge.",
        "Plato described reality as shadows on a cave wall.",
        "The Scientific Revolution overturned Aristotelian cosmology.",
        "Godel showed arithmetic cannot be both complete and consistent.",
        "The Enlightenment championed reason and individual rights.",
        "Darwin proposed natural selection as the mechanism of evolution.",
        "The Cold War was defined by nuclear deterrence strategies.",
        "Kant argued morality derives from pure reason.",
        "The Industrial Revolution mechanised production and urbanised society.",
        "Nietzsche proclaimed the death of God and the rise of the will to power.",
    ]


# ── Audio ─────────────────────────────────────────────────────────────────────

class AudioEmbedder:
    """
    Encode audio files into L2-normalised embeddings using librosa MFCCs.

    This is a lightweight approach that doesn't require large model downloads:
    extract MFCCs (mel-frequency cepstral coefficients), take temporal statistics,
    and L2-normalise. Output is a fixed-size vector regardless of audio length.

    Parameters
    ----------
    n_mfcc  : int -- number of MFCC coefficients (default 40)
    sr      : int -- target sample rate in Hz (default 22050)
    """

    def __init__(self, n_mfcc=40, sr=22050):
        try:
            import librosa  # noqa: F401
        except ImportError:
            raise ImportError(
                "librosa not found. Run: pip install librosa"
            )
        self.n_mfcc = n_mfcc
        self.sr = sr

    def _extract_features(self, audio_path):
        """Extract a fixed-size feature vector from an audio file."""
        import librosa
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        # Temporal statistics: mean, std, delta-mean, delta-std
        delta = librosa.feature.delta(mfccs)
        features = np.concatenate([
            mfccs.mean(axis=1), mfccs.std(axis=1),
            delta.mean(axis=1), delta.std(axis=1),
        ])
        return features

    def encode(self, audio_paths, show_progress=False):
        """
        Encode a list of audio file paths into an (N, 160) float32 array.
        Vectors are L2-normalised.

        Parameters
        ----------
        audio_paths  : list[str]  -- paths to audio files (.wav, .mp3, etc.)
        show_progress: bool

        Returns
        -------
        embeddings : np.ndarray, shape (N, n_mfcc*4), float32, L2-normalised
        """
        vecs = []
        iterator = enumerate(audio_paths)
        for i, path in iterator:
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Audio: {i+1}/{len(audio_paths)}")
            vecs.append(self._extract_features(path))
        X = np.array(vecs, dtype=np.float32)
        # L2-normalise
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        return X

    @property
    def dim(self):
        return self.n_mfcc * 4  # mean + std + delta_mean + delta_std

    def __repr__(self):
        return f"AudioEmbedder(n_mfcc={self.n_mfcc}, sr={self.sr}, dim={self.dim})"
