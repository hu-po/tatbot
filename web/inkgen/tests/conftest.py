"""The inkgen app modules (app, batch, engine, ...) import by bare name from web/inkgen."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
