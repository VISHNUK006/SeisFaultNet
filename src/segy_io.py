"""Optional SEG-Y reader. Install segyio separately when working with real SEG-Y files."""
from pathlib import Path
import tempfile
import numpy as np

def read_segy_bytes(raw_bytes):
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError(
            "SEG-Y support is optional. Install it with: pip install -r requirements-segyio.txt"
        ) from exc
    with tempfile.NamedTemporaryFile(suffix=".sgy",delete=False) as f:
        f.write(raw_bytes); path=f.name
    try:
        with segyio.open(path,"r",ignore_geometry=True) as f:
            return np.asarray(segyio.tools.cube(f),dtype=np.float32)
    finally:
        Path(path).unlink(missing_ok=True)
