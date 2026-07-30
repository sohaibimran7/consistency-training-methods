"""Offline reproduction adapters for Irpan et al. (arXiv:2510.27062)."""

from ctm_data.adapters.irpan_2510_27062.schema import ARTIFACT_SCHEMA, PAPER_ID, SCHEMA_VERSION
from ctm_data.adapters.irpan_2510_27062.partitions import ARTIFACT_ROLES, PARTITION_REGISTRY
from ctm_data.adapters.irpan_2510_27062.source_registry import SOURCES, SourceSpec, require_source

__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_ROLES",
    "PAPER_ID",
    "PARTITION_REGISTRY",
    "SCHEMA_VERSION",
    "SOURCES",
    "SourceSpec",
    "require_source",
]
