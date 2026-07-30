"""Compatibility imports for the shared generation-provenance contract.

The Irpan reproduction consumes these generic identities, but their schema is
owned by :mod:`ctm.generation_provenance` so other training pipelines can use
the same fail-closed freshness checks without depending on paper code.
"""

from ctm import generation_provenance as _generation_provenance
from ctm.generation_provenance import *

__all__ = _generation_provenance.__all__
