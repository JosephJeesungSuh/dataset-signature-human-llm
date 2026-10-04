"""Paths, dataset order, and experiment definitions shared by all scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("DSIG_ARTIFACTS", ROOT / "artifacts"))

DATA = ARTIFACTS / "data"
INTERIM = DATA / "interim"                  # English, structurally valid records per source
CORPUS_ORIGINAL = DATA / "corpus_original"  # selected conversations with dataset-specific markers kept
CORPUS = DATA / "corpus"                    # same conversations with markers deleted (main input)
SYNTHETIC = DATA / "synthetic"              # user-model x assistant conversations (Sec. 3.1, App. E)
EMBEDDINGS = ARTIFACTS / "embeddings"
RESULTS = ARTIFACTS / "results"
USERLM_DATA = ARTIFACTS / "userlm_data"     # per-source processed data, intents, and SFT samples
MODELS = ARTIFACTS / "models"
LID_MODEL = ARTIFACTS / "lid.176.bin"       # fastText language identification model

SOURCES = ["wildchat_1m", "wildchat_4p8m", "lmsys", "arena_2025", "sharechat", "sharegpt", "hh_rlhf"]
DISPLAY = {"wildchat_1m": "WC-1M", "wildchat_4p8m": "WC-4.8M", "lmsys": "LMSYS", "arena_2025": "Arena",
           "sharechat": "ShareChat", "sharegpt": "ShareGPT", "hh_rlhf": "HH-RLHF"}
SPLITS = ("train", "val", "test")
SPLIT_SIZES = {"train": 40000, "val": 3000, "test": 7000}
SEED = 42

ASSISTANT_MASK = "<assistant_response>"
USER_MASK = "<user_query>"

# Frozen representations (App. B): layers read from each decoder, mean and final-EOS pooling.
QWEN_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [6, 12, 18, 24],
    "Qwen/Qwen2.5-1.5B": [6, 12, 18, 28],
    "Qwen/Qwen2.5-3B": [9, 18, 27, 36],
    "Qwen/Qwen2.5-7B": [6, 12, 18, 28],
    "Qwen/Qwen2.5-14B": [12, 24, 36, 48],
}
EMBED_MODEL = "Qwen/Qwen2.5-7B"
EMBED_MAX_LEN = 32768
# Encoders from other model families (Fig. 3): model, context length, pooling.
ENCODERS = {
    "bert_tiny": ("google/bert_uncased_L-2_H-128_A-2", 512, "cls"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 512, "mean_normalized"),
    "deberta": ("microsoft/deberta-v3-small", 512, "cls"),
    "modernbert": ("answerdotai/ModernBERT-base", 8192, "cls"),
}

# Sec. 2.2, Table 2.
BINARY_PAIRS = [("wildchat_1m", "wildchat_4p8m"), ("lmsys", "arena_2025"), ("sharechat", "sharegpt")]
FOCAL = ["wildchat_1m", "lmsys", "sharechat"]
TERNARY = [FOCAL,
           ["wildchat_1m", "lmsys", "hh_rlhf"],
           ["wildchat_1m", "wildchat_4p8m", "sharegpt"],
           ["wildchat_4p8m", "lmsys", "hh_rlhf"],
           ["wildchat_4p8m", "lmsys", "sharechat"]]
INCREMENTAL = [SOURCES[:k] for k in range(3, len(SOURCES) + 1)]

# Sec. 2.3.
TAXONOMY_SOURCES = ["wildchat_1m", "lmsys", "sharegpt"]

# Sec. 3: user-model base models and the intent generator.
USER_BASE_MODELS = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3-8B"]
INTENT_MODEL = "Qwen/Qwen3-32B"

# Sec. 3.1 / App. E.2: (user model A, user model B, intent source).
SYNTHETIC_PAIRS = [
    ("wildchat_1m", "wildchat_4p8m", "hh_rlhf"),
    ("wildchat_4p8m", "lmsys", "wildchat_1m"),
    ("wildchat_4p8m", "sharechat", "wildchat_1m"),
    ("lmsys", "sharechat", "wildchat_4p8m"),
    ("lmsys", "sharegpt", "wildchat_4p8m"),
    ("sharechat", "sharegpt", "lmsys"),
    ("sharechat", "arena_2025", "lmsys"),
    ("sharegpt", "arena_2025", "sharechat"),
    ("sharegpt", "hh_rlhf", "sharechat"),
    ("arena_2025", "hh_rlhf", "sharegpt"),
]
# App. E.3: (intent source, user model) combinations conversing with every assistant.
ASSISTANT_IDENTITY_COMBOS = [
    ("hh_rlhf", "wildchat_1m"), ("hh_rlhf", "wildchat_4p8m"), ("hh_rlhf", "lmsys"),
    ("wildchat_1m", "wildchat_4p8m"), ("wildchat_1m", "lmsys"), ("wildchat_1m", "sharechat"),
    ("wildchat_4p8m", "lmsys"), ("wildchat_4p8m", "sharechat"), ("wildchat_4p8m", "sharegpt"),
    ("lmsys", "sharechat"), ("lmsys", "sharegpt"), ("lmsys", "arena_2025"),
    ("sharechat", "sharegpt"), ("sharechat", "arena_2025"), ("sharechat", "hh_rlhf"),
    ("sharegpt", "arena_2025"), ("sharegpt", "hh_rlhf"),
]
# Assistants used with the synthetic conversations: short tag -> served model.
ASSISTANTS = {
    "qwen3p5-9b": "Qwen/Qwen3.5-9B",
    "llama3p1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2p5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
}


def model_tag(model):
    """Filesystem-safe model name, e.g. Qwen/Qwen2.5-7B-Instruct -> Qwen--Qwen2.5-7B-Instruct."""
    return model.replace("/", "--")


def user_family(base_model):
    return "llama3" if "llama" in base_model.lower() else "qwen2.5"


def synthetic_name(intent_source, user_source, assistant_tag, base_model="meta-llama/Meta-Llama-3-8B"):
    """Directory name of one synthetic conversation set under SYNTHETIC."""
    return f"synth_{intent_source}--user_{user_source}_{user_family(base_model)}--assistant_{assistant_tag}"
