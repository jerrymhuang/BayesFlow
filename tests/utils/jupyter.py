import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from pathlib import Path


def run_notebook(path):
    with open(str(path)) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    kernel = ExecutePreprocessor(
        timeout=600, kernel_name="python3", resources={"metadata": {"path": Path(path).parent}}
    )

    result = kernel.preprocess(nb)
    return result
