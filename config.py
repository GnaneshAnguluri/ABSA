# ─────────────────────────────────────────────
#  ABSA Pipeline — Configuration File
# ─────────────────────────────────────────────

# --- Dataset ---
DATASET_NAME   = "aste"          # Options: "aste", "semeval14"
DATASET_SPLIT  = "14res"         # Options: "14res", "14lap", "15res", "16res"
DATA_DIR       = "data/raw"
PROCESSED_DIR  = "data/processed"

# --- Model ---
ENCODER_MODEL  = "bert-base-uncased"   # or "roberta-base"
MAX_SEQ_LEN    = 128
HIDDEN_SIZE    = 768               # BERT base hidden size

# --- BIO Tag IDs ---
ASPECT_TAGS    = {"O": 0, "B-ASP": 1, "I-ASP": 2}
OPINION_TAGS   = {"O": 0, "B-OPN": 1, "I-OPN": 2}
SENTIMENT_MAP  = {"NEG": 0, "NEU": 1, "POS": 2}
ID2SENTIMENT   = {0: "NEG", 1: "NEU", 2: "POS"}

# --- Training ---
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
EPOCHS         = 20
WARMUP_STEPS   = 100
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0

# --- Loss Weights (λ1, λ2, λ3) ---
LAMBDA_ASPECT   = 1.0
LAMBDA_OPINION  = 1.0
LAMBDA_SENTIMENT= 1.0

# --- Paths ---
CHECKPOINT_DIR  = "checkpoints"
LOG_DIR         = "logs"

# --- Reproducibility ---
SEED            = 42
