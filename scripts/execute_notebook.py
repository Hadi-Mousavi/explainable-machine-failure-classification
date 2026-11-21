"""Execute the thin Day 1 notebook with this exact Python environment."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main() -> None:
    """Use a temporary kernelspec, avoiding global Jupyter registration changes."""
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/01_eda.ipynb"
    output = root / "notebooks/01_eda_executed.ipynb"
    notebook = nbformat.read(source, as_version=4)
    with TemporaryDirectory(prefix="ai4i-kernel-") as directory:
        spec = Path(directory) / "ai4i-day1"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "AI4I Day 1 (current environment)",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name="ai4i-day1",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=300,
            allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.write(notebook, output)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not all(cell.execution_count is not None for cell in code_cells):
        raise RuntimeError("Not all notebook code cells executed")
    print(f"Executed {len(code_cells)} code cells without errors: {output}")


if __name__ == "__main__":
    main()
