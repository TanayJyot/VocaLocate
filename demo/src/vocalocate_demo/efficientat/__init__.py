"""MobileNetV3 for audio from EfficientAT (github.com/fschmid56/EfficientAT, MIT licence, see LICENSE).

Vendored from models/mn at commit a425fdce92572e602a1d5634799bd9f1f2efa806 so the demo can load the
AudioSet-pretrained weights the QVIM challenge baseline uses (mn10_as). Changes: package-relative imports,
a local ConvNormActivation instead of torchvision's, NAME_TO_WIDTH inlined, weights cached under
~/.cache/efficientat (or $EFFICIENTAT_CACHE), and no model print on construction.
"""

from .model import get_model

__all__ = ["get_model"]
