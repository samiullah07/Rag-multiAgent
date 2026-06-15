# scripts/print_routes.py
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.main import app  # now this should work

for route in app.routes:
    methods = ",".join(sorted(route.methods))
    print(f"{methods:15} {route.path}")