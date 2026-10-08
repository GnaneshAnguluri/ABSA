# ─────────────────────────────────────────────
#  ABSA / ASTE Pipeline — Configuration File
# ─────────────────────────────────────────────

# --- Dataset ---
DATASET_NAME   = "aste"          # Options: "aste", "semeval14"
DATASET_SPLIT  = "14res"         # Options: "14res", "14lap", "15res", "16res"
DATA_DIR       = "data/raw"
PROCESSED_DIR  = "data/processed"

# --- Model & Encoder ---
ENCODER_MODEL         = "bert-base-uncased"
MAX_SEQ_LEN           = 128
HIDDEN_SIZE           = 768               # BERT base hidden size
PAIR_HIDDEN_SIZE      = 1536              # 768 * 2 (concatenated aspect + opinion vectors)
FREEZE_ENCODER        = True              # Phase 1: Freeze BERT parameters
USE_CACHED_EMBEDDINGS = True              # CPU Optimization: Cache frozen BERT embeddings

# --- Label Counts & Tag Mappings ---
NUM_ASPECT_LABELS   = 3             # O: 0, B-ASP: 1, I-ASP: 2
NUM_OPINION_LABELS  = 3             # O: 0, B-OPN: 1, I-OPN: 2
NUM_RELATION_LABELS = 2             # INVALID: 0, VALID: 1
NUM_SENTIMENT_LABELS= 3             # NEG: 0, NEU: 1, POS: 2

ASPECT_TAGS   = {"O": 0, "B-ASP": 1, "I-ASP": 2}
OPINION_TAGS  = {"O": 0, "B-OPN": 1, "I-OPN": 2}
RELATION_MAP  = {"INVALID": 0, "VALID": 1}
SENTIMENT_MAP = {"NEG": 0, "NEU": 1, "POS": 2}
ID2SENTIMENT  = {0: "NEG", 1: "NEU", 2: "POS"}

# --- Training Hyperparameters ---
BATCH_SIZE     = 8                  # Set to 8 for CPU optimization
LEARNING_RATE  = 1e-3               # LR for ASTE linear heads in Phase 1
WEIGHT_DECAY   = 0.01
EPOCHS         = 5                  # Phase 1: 5 epochs
WARMUP_STEPS   = 50
MAX_GRAD_NORM  = 1.0

# --- Multi-Task Loss Weights (λ1, λ2, λ3, λ4) ---
LAMBDA_ASPECT    = 1.0
LAMBDA_OPINION   = 1.0
LAMBDA_RELATION  = 1.0
LAMBDA_SENTIMENT = 1.0

# --- Paths ---
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"

# --- Reproducibility ---
SEED           = 42
