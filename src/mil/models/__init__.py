"""
Model classes are defined in train_mm_abmil_v7.py and importable via the
package once src/ is on the Python path.  Use build_model_v7() as the
canonical factory — it handles all fusion variants.

Example
-------
>>> import sys; sys.path.insert(0, "/path/to/chicago_mil/src")
>>> from mil.models.builders import build_model_v7
>>> model = build_model_v7("middle")

The full extraction of model classes into submodules is tracked in the
project roadmap and will be completed incrementally to avoid breaking
existing SLURM jobs.
"""
