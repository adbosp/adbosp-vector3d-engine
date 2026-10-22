"""Pipeline stage registry.

Every stage is an independent module with a typed input and output, so agents
(Claude, Codex, ...) can call any stage directly through the API/MCP layer.
`needs_llm` marks stages where a model may be used; all others must be
solved deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    module: str
    needs_llm: bool
    implemented: bool


STAGES: tuple[StageSpec, ...] = (
    StageSpec("parse", "SVG Geometry Parser", "vector3d.svg.parser", False, True),
    StageSpec("normalize", "Coordinate Normalization", "vector3d.normalize.normalizer", False, True),
    StageSpec("semantics", "AI Semantic Geometry Understanding", "vector3d.semantics", True, False),
    StageSpec("correspond", "Cross-view Correspondence Solver", "vector3d.correspond", False, False),
    StageSpec("reconstruct", "Confidence-aware 3D Reconstruction", "vector3d.reconstruct", False, False),
    StageSpec("coarse_surface", "Coarse 3D Surface Reconstruction", "vector3d.reconstruct.hull", False, True),
    StageSpec("topology_blueprint", "AI SVG Topology Blueprint", "vector3d.topology.blueprint", True, False),
    StageSpec("topology_compile", "Topology Graph Compiler", "vector3d.topology.compiler", False, False),
    StageSpec("surface_fit", "3D Surface Fitting Solver", "vector3d.surface.fit", False, False),
    StageSpec("validate", "Geometry + Topology Validation", "vector3d.reconstruct.surface", False, True),
    StageSpec("refine", "Iterative Refinement", "vector3d.refine", True, False),
    StageSpec("export", "Export OBJ / GLB / Metadata", "vector3d.export.writers", False, True),
)
