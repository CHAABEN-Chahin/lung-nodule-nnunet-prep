import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(BASE_DIR)

# NIfTI C library cannot handle Unicode paths on Windows (the 'é' in
# P2M_pipeline_terminé breaks SimpleITK writes).  Keep all heavy I/O
# under a plain-ASCII directory.
NODULIS_HOME = os.path.join(os.path.expanduser("~"), "nodulis_data")
DATA_DIR = os.path.join(NODULIS_HOME, "cases")
DB_PATH = os.path.join(NODULIS_HOME, "nodulis.db")
TEMP_DIR = os.path.join(NODULIS_HOME, "tmp")

# Model files are only *read* by nnU-Net subprocess — copy once to a
# safe path so the RESULTS_FOLDER env var is also ASCII-clean.
_SRC_MODEL = os.path.join(PIPELINE_ROOT, "model")
MODEL_DIR = os.path.join(NODULIS_HOME, "model")

_SRC_CLASSIFIER = os.path.join(
    PIPELINE_ROOT, "nodule_filter_classifier", "nodule_classifier.state"
)
CLASSIFIER_PATH = os.path.join(NODULIS_HOME, "nodule_classifier.state")

SECRET_KEY = "nodulis-local-2026"


def ensure_paths():
    """Create dirs and copy model files on first run."""
    for d in (DATA_DIR, TEMP_DIR):
        os.makedirs(d, exist_ok=True)

    if not os.path.isdir(MODEL_DIR) and os.path.isdir(_SRC_MODEL):
        print(f"  Copying model files to {MODEL_DIR} ...")
        shutil.copytree(_SRC_MODEL, MODEL_DIR)

    if not os.path.isfile(CLASSIFIER_PATH) and os.path.isfile(_SRC_CLASSIFIER):
        print(f"  Copying classifier weights to {CLASSIFIER_PATH} ...")
        shutil.copy2(_SRC_CLASSIFIER, CLASSIFIER_PATH)
