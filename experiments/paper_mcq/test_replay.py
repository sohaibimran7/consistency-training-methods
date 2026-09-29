"""Fast contract tests; these do not claim a full statistical replay."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('replay',Path(__file__).with_name('run.py'))
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

class ReplayContract(unittest.TestCase):
    def test_source_lock(self):
        self.assertGreater(len(replay.verify_code()),15)

    def test_generator_lock(self):
        lock=json.loads((replay.BUNDLE/'generator-source-lock.json').read_text())
        for name,item in lock.items():
            self.assertEqual(replay.sha(replay.BUNDLE/name),item['sha256'])

    def test_bundled_python_compiles(self):
        for directory in ('recipes','vendor','generators'):
            for path in (replay.BUNDLE/directory).rglob('*.py'):
                compile(path.read_text(),str(path),'exec')

    def test_anchor_fails_closed(self):
        for text in ('missing','aa'):
            with self.assertRaises(ValueError):
                replay.replace_once(text,'a','b')
        self.assertEqual(replay.replace_once('cat','cat','dog'),'dog')

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                replay.fresh_output(Path(tmp),[])

    def test_no_input_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                replay.fresh_output(Path(tmp)/'out',[Path(tmp)])

    def test_environment_preserves_permutations(self):
        env=replay.environment(Path('/vendor'),Path('/output'))
        self.assertEqual(env['CTM_CONDITIONAL_PERMUTATIONS'],'108000')
        self.assertEqual(env['OPENBLAS_NUM_THREADS'],'1')
        self.assertEqual(env['PYTHONPATH'],'/vendor')

    def test_isolated_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor=replay.stage_vendor(Path(tmp))
            self.assertTrue((vendor/'ctm_data/adapters/mcq_bias/__init__.py').is_file())
            self.assertTrue((vendor/'mcq_bias/parsers.py').is_file())
            self.assertNotIn('setting', (vendor/'ctm_data/adapters/mcq_bias/__init__.py').read_text())

if __name__=='__main__':
    unittest.main()
