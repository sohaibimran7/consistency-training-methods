"""Contract tests for offline assembly; fixtures contain no research data."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import build_figures as build


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root/'inputs'; self.inputs.mkdir()
        self.output = self.root/'output'
        self.manifest = self.root/'sources.json'
        self.doc = {'schema':'ctm-manuscript-figure-sources-v1','scope':'test fixtures',
                    'sources':{},'entries':[]}
        style = {'condition_order':list(build.LABELS),
                 'condition_styles':{k:{'color':'gray'} for k in build.LABELS}}
        rows = [dict(population=p,bias_type=b,condition=m,mean=.5,ci_lower=.2,
                     ci_upper=.8,n_scored=10) for p in ('held_in_datasets','held_out_dataset')
                for b in ('seen_mean','held_out_mean') for m in build.LABELS]
        self.source('style',json.dumps(style).encode())
        for key in ('main-switch','main-conditional'):
            self.source(key,json.dumps(rows).encode())
            self.doc['entries'].append(dict(id=key,kind='render_saved_statistics',
                sources=[key,'style'],output=f'figures/{key}.pdf',status='provisional'))
        self.source('copy',b'saved figure fixture')
        self.doc['entries'].append(dict(id='copy',kind='copy_saved_figure',
            sources=['copy'],output='figures/copied.pdf',status='historical'))
        self.source('table',b'saved accounting fixture')
        self.doc['entries'].append(dict(id='table',kind='source_only_manual_table',
            sources=['table'],output=None,status='not generated'))

    def source(self,key,data):
        (self.inputs/key).write_bytes(data)
        self.doc['sources'][key] = dict(root='ctm_artifacts',path=key,
            sha256=hashlib.sha256(data).hexdigest())

    def run_builder(self,check=True):
        self.manifest.write_text(json.dumps(self.doc))
        argv=['--manifest',str(self.manifest),'--output-dir',str(self.output)]
        for name in build.ROOT_NAMES:
            argv += ['--'+name.replace('_','-')+'-root',str(self.inputs)]
        if check: argv += ['--check-only']
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return build.main(argv)

    def rejected_without_outputs(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_builder(check=False)
        self.assertEqual(raised.exception.code,2)
        self.assertFalse(self.output.exists())

    def test_check_only_has_no_output(self):
        self.run_builder()
        self.assertFalse(self.output.exists())

    def test_late_source_tamper_fails_before_any_output(self):
        (self.inputs/'table').write_bytes(b'tampered')
        self.rejected_without_outputs()

    def test_duplicate_cell_rejected_even_with_matching_hash(self):
        rows=json.loads((self.inputs/'main-switch').read_bytes())
        rows.append(rows[0])
        self.source('main-switch',json.dumps(rows).encode())
        self.rejected_without_outputs()

    def test_undefined_cell_is_not_imputed(self):
        rows=json.loads((self.inputs/'main-switch').read_bytes())
        rows[0]['mean']=None
        self.source('main-switch',json.dumps(rows).encode())
        self.rejected_without_outputs()

    def test_output_escape_rejected(self):
        self.doc['entries'][2]['output']='../outside.pdf'
        self.rejected_without_outputs()

    def test_output_collision_rejected(self):
        self.doc['entries'][2]['output']='figures/main-switch.png'
        self.rejected_without_outputs()

    def test_copy_identity_and_tables_not_claimed_generated(self):
        # The full real-data smoke run separately exercises Matplotlib rendering.
        self.doc['entries']=self.doc['entries'][2:]
        self.run_builder(check=False)
        self.assertEqual((self.output/'figures/copied.pdf').read_bytes(),b'saved figure fixture')
        inventory=json.loads((self.output/'figure-inventory.json').read_text())
        self.assertEqual(inventory['entries'][1]['outputs'],[])
        self.assertFalse(inventory['upstream_aggregation_performed'])
        self.assertEqual(inventory['entries'][0]['outputs'][0]['sha256'],
                         self.doc['sources']['copy']['sha256'])


if __name__ == '__main__':
    unittest.main()
