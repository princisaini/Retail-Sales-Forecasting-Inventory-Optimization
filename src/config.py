from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "images"

for folder in (DATA_RAW, DATA_PROCESSED, MODELS, OUTPUTS, IMAGES):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FORECAST_HORIZON = 7