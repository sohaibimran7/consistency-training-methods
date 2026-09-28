"""Offline reproduction adapters for Irpan et al. (arXiv:2510.27062)."""

from scripts.irpan_2510_27062.partitions import ARTIFACT_ROLES, PARTITION_REGISTRY
from scripts.irpan_2510_27062.schema import ARTIFACT_SCHEMA, PAPER_ID, SCHEMA_VERSION
from scripts.irpan_2510_27062.source_registry import SOURCES, SourceSpec, require_source

__all__ = [
    "ARTIFACT_ROLES",
    "ARTIFACT_SCHEMA",
    "PAPER_ID",
    "PARTITION_REGISTRY",
    "SCHEMA_VERSION",
    "SOURCES",
    "SourceSpec",
    "require_source",
]
