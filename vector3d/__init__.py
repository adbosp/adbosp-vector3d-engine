"""Vector3D Engine: orthographic SVG views -> verified 3D geometry."""
from .core.confidence import Confidence
from .normalize.normalizer import NormalizedScene, normalize_views
from .normalize.views import CAMERAS, OPPOSITE, ViewName
from .svg.parser import SvgDocument, parse_svg

__version__ = "0.3.0"
__all__ = ["Confidence", "NormalizedScene", "normalize_views", "CAMERAS", "OPPOSITE", "ViewName",
           "SvgDocument", "parse_svg"]
