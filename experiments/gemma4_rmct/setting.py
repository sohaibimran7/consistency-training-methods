"""Continue the frozen Qwen QID permutation without a 32-window ceiling."""

import copy
from ctm_data.adapters.mcq_bias.shared_qid_two_bias import (
    SharedQidTwoBiasSetting, DATASETS, QIDS_PER_DATASET,
    _validated_manifest_permutation, _ids_sha256,
)


class ContinuingSharedQidSetting(SharedQidTwoBiasSetting):
    def load_datapoints(self, n_datapoints=32, *, segment_index=0, batch_offset=None, **kwargs):
        if batch_offset is None:
            return super().load_datapoints(n_datapoints, segment_index=segment_index, **kwargs)
        if type(batch_offset) is not int or batch_offset < 0:
            raise ValueError("batch_offset must be a nonnegative attempted-batch counter")
        if type(n_datapoints) is not int or n_datapoints < 2 or n_datapoints > 32 or n_datapoints % 2:
            raise ValueError("recovery slices require 1..16 whole two-QID batches")
        manifest, rows = self._load_verified()
        count = n_datapoints // 2
        selected, metadata = {}, {}
        for dataset in DATASETS:
            permutation = _validated_manifest_permutation(manifest, dataset=dataset)
            indices = [(batch_offset+i) % QIDS_PER_DATASET for i in range(count)]
            ids = [permutation[i] for i in indices]
            selected[dataset] = ids
            metadata[dataset] = dict(permutation_offset=batch_offset, permutation_indices=indices,
                question_ids=ids, question_ids_sha256=_ids_sha256(ids))
        ids = [selected[dataset][i] for i in range(count) for dataset in DATASETS]
        self._loaded_segment = dict(segment_index=segment_index, batch_offset=batch_offset,
            n_datapoints=n_datapoints, attempted_batches=count,
            selection="cyclic_contiguous_per_dataset_permutation_then_logiqa_hellaswag_interleave",
            per_dataset=metadata, interleaved_question_ids=ids, interleaved_question_ids_sha256=_ids_sha256(ids))
        return [copy.deepcopy(rows[qid]) for qid in ids]

    @staticmethod
    def _validate_segment_request(n_datapoints, segment_index):
        # Retain the parent's exact window size and cyclic selection. Only
        # relax the legacy run-length bound, preserving the real global index.
        SharedQidTwoBiasSetting._validate_segment_request(n_datapoints, 0)
        if type(segment_index) is not int or segment_index < 0:
            raise ValueError("segment_index must be a nonnegative integer")
        return n_datapoints, segment_index


def create_setting(**kwargs):
    return ContinuingSharedQidSetting(**kwargs)
