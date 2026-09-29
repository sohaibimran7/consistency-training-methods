"""Integration checks using existing reference logs; never call a grader API."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.rmct_two_bias_eval import step352_postprocess as publication


class PostprocessTests(unittest.TestCase):
    def test_full_pool_bars_with_matched_significance(self):
        sources, _ = publication.reference_sources("towards_bias_switch")
        # A deliberate self-comparison fixture, confined to a temporary directory.
        with TemporaryDirectory(prefix="ctm352-SYNTHETIC-TEST-") as temporary:
            root = Path(temporary)
            with patch.object(publication, "new_sources", return_value=sources["rmct_step16"]), \
                 patch("ctm_data.adapters.mcq_bias.plot.render_publication_plot") as render:
                publication.plot(root, root, "towards_bias_switch")
                self.assertEqual(render.call_count, 4)
            for reference in publication.REFERENCES:
                folder = root / "plots" / f"towards_bias_switch-vs-{reference}"
                rows = json.loads((folder / "chart-rows.json").read_text())
                self.assertEqual(len(rows), 54)
                cells = {(row["condition"], row["population"], row["bias_type"]): row for row in rows}
                for condition, count in [("rmct_step16", 100), ("rmct_step176", 200), ("rmct_step352", 100)]:
                    self.assertEqual(cells[(condition, "held_in_datasets", "wrong_argument")]["n_total"], count)
                tested = [row for row in rows if row["condition"] == publication.NEW]
                self.assertEqual(len(tested), 18)
                self.assertTrue(all(row["holm_family_size"] == 36 for row in tested))
                self.assertTrue(all(row["question_clusters"] == 100 for row in tested))
                self.assertTrue(all(row["significance_baseline"] == reference for row in tested))
                if reference == "rmct_step16":
                    self.assertTrue(all(row["p_value"] == 1 and row["observed_difference"] == 0 for row in tested))

    def test_reference_hashes_for_both_metrics(self):
        for metric in ("towards_bias_switch", "bias_acknowledged"):
            with self.subTest(metric=metric):
                sources, _ = publication.reference_sources(metric)
                self.assertEqual({key: len(value) for key, value in sources.items()},
                                 {"rmct_step16": 18, "rmct_step176": 18})
