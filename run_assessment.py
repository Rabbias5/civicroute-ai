"""Run the assessment notebook's code cells from the project directory."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")

os.chdir(Path(__file__).resolve().parent)
notebook = json.loads(Path("civicroute_assessment.ipynb").read_text())
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        print(f"\nRunning notebook cell {index}")
        code = "".join(cell["source"])
        exec(compile(code, f"notebook_cell_{index}", "exec"), namespace)
print("\nFinished. Evidence is saved in outputs/.")
