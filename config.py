
import os

class Config:
    SEQ_LEN = 50
    PRED_LEN = 1
    TEST_RATIO = 0.2
    VAL_RATIO = 0.1

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-3
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    HIDDEN_DIM = 128

    SEED = 42
    SAVE_DIR = "checkpoints"
    REPORTS_DIR = "reports"
