"""VocaLocate data-processing pipeline.

Raw audio from public datasets, team recordings and a user's own SFX folders
moves through three lake layers (bronze -> silver -> gold) and is loaded into a
DuckDB warehouse. See pipeline/README.md for the design.
"""

__version__ = "0.1.0"
