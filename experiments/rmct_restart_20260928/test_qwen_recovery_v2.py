"""CPU controller tests. Synthetic tensor files do not establish GPU loadability."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rmct_restart_20260928 import qwen_train_window as w
from experiments.rmct_restart_20260928.qwen_checkpoint import FILES, identity, checkpoint_state, seal_v2
from experiments.rmct_restart_20260928.qwen_progress import progress, next_slice


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.repo = self.root / 'new'
        self.plan = json.loads((Path(__file__).parent / 'fixtures/original_recipe.json').read_text())
        a = self.plan['argv']; i = a.index('--resume-from'); del a[i:i+2]
        a.remove('--resume-with-optimizer'); a.remove('--resume-state-required')
        a[1] = str(self.repo / 'scripts/train_rlct.py')
        a[a.index('--load-config')+1] = json.dumps(dict(n_datapoints=32, segment_index=0, cycle_segments=True))
        self.plan.update(incorporated_commit='b'*40, python_prefix='/unchanged/python')
        self.plan_path = self.root / 'plan.json'
        self.bind()

    def bind(self):
        save(self.plan_path, self.plan)
        self.binding = w.plan_binding(self.plan, self.plan_path)

    def checkpoint(self, path, before, count, updates):
        path.mkdir(parents=True)
        for name in FILES:
            (path/name).write_bytes(b'synthetic-test-file-not-real-tensors')
        rng = dict(schema='ctm.rl_runtime_rng.v1', python_random_state=[],
                   torch_cpu_rng_state_base64='AA==', torch_cuda_rng_state_base64='AA==',
                   torch_cuda_coordinator_device=0)
        batches = before['sampled_batches'] + count
        loop = dict(schema='ctm.rl_loop_state.v1', epoch=1, step=batches, global_step=batches,
                    optimizer_step=updates, completed_epochs=1, segment_start_global_step=before['sampled_batches'],
                    segment_step=count, accumulated_grads=0, final=True, runtime_rng=rng)
        save(path/'manifest.json', dict(backend='local', kind='both', loop_state=loop))
        save(path/'replicated_training_manifest.json', dict(checkpoint_kind='both', world_size=4,
             train_logical_indices=[0,1,2,3], process_group_backend='nccl', device_type='cuda',
             rng_state_file='replicated_training_rng.pt'))
        return loop

    def fresh_seal(self, count=16, updates=12):
        before=progress(0,0); selection=dict(segment_index=0,batch_offset=0,batch_count=count)
        argv=w.command_v2(self.plan,before,selection,None)
        cp=w.checkpoint_path(argv,self.repo);self.checkpoint(cp,before,count,updates)
        return seal_v2(cp,campaign_id=self.plan['argv'][self.plan['argv'].index('--run-name')+1],
                       before=before,selection=selection,command=argv,binding=self.binding)

    def test_unequal_counters_seal_and_reverify(self):
        receipt=self.fresh_seal()
        self.assertEqual(receipt['step'],12)
        self.assertEqual(receipt['progress'],progress(16,12))
        self.assertEqual(w.verify_seal_v2(receipt,self.plan,self.binding),receipt)
        self.assertFalse(receipt['bitwise_rollout_continuation'])

    def test_partial_or_overshot_checkpoint_rejected(self):
        receipt=self.fresh_seal()
        path=Path(receipt['checkpoint'])/'manifest.json';original=json.loads(path.read_text())
        for key,value in [('final',False),('optimizer_step',17),('segment_step',15),('segment_start_global_step',1),('accumulated_grads',1)]:
            changed=copy.deepcopy(original);changed['loop_state'][key]=value;save(path,changed)
            with self.assertRaises(ValueError):checkpoint_state(path.parent,progress(0,0),receipt['selection'])
        save(path,original)

    def test_changed_bytes_or_receipt_rejected_before_child(self):
        receipt=self.fresh_seal()
        for mutation in ('step','command','binding','progress'):
            bad=copy.deepcopy(receipt)
            if mutation=='step':bad['step']=16
            elif mutation=='command':bad['command'].append('--different-recipe')
            elif mutation=='binding':bad['binding']['source_commit']='c'*40
            else:bad['progress']['sampled_batches']=12
            with patch.object(w.subprocess,'run') as launch:
                with self.assertRaises(ValueError):
                    w.train_to_boundary(self.plan,self.binding,self.root/'campaign',64,{},bad,{})
                launch.assert_not_called()
        (Path(receipt['checkpoint'])/'optimizer.pt').write_bytes(b'changed')
        with patch.object(w.subprocess,'run') as launch:
            with self.assertRaises(ValueError):w.train_to_boundary(self.plan,self.binding,self.root/'campaign',64,{},receipt,{})
            launch.assert_not_called()

    def test_dynamic_window_skips_no_replay_and_exact_boundary(self):
        seen=[]
        def train(argv,**kwargs):
            load=json.loads(argv[argv.index('--load-config')+1]); start=load['segment_index']*16+load['batch_offset']
            count=load['batch_count'];updates=0
            if '--resume-from' in argv:
                old=Path(argv[argv.index('--resume-from')+1][7:])
                updates=json.loads((old/'manifest.json').read_text())['loop_state']['optimizer_step']
            # First child skips four updates, last full child skips two more.
            delta=count-(4 if start==0 else 2 if start==64 else 0)
            self.checkpoint(w.checkpoint_path(argv,self.repo),progress(start,updates),count,updates+delta)
            seen.extend(range(start,start+count))
        with patch.object(w,'verify_data_slice',return_value=[]),patch.object(w.subprocess,'run',side_effect=train):
            final=w.train_to_boundary(self.plan,self.binding,self.root/'campaign',64,{},None,{})
        self.assertEqual(final['step'],64)
        self.assertEqual(final['progress']['sampled_batches'],70)
        self.assertEqual(seen,list(range(70)))
        self.assertEqual(final['selection']['batch_count'],2)
        self.assertEqual(w.verify_seal_v2(final,self.plan,self.binding),final)

    def test_uncertain_start_not_replayed(self):
        before=progress(0,0);selection=next_slice(before,64)
        folder=self.root/'campaign'/'batches'/'0';save(folder/'started.json',{'uncertain':True})
        with patch.object(w,'verify_data_slice',return_value=[]),patch.object(w.subprocess,'run') as launch:
            with self.assertRaises(FileExistsError):w.train_to_boundary(self.plan,self.binding,self.root/'campaign',64,{},None,{})
            launch.assert_not_called()

    def recovery_fixture(self):
        original=copy.deepcopy(self.plan);original['incorporated_commit']='a'*40
        old_repo=self.root/'old';original['argv'][1]=str(old_repo/'scripts/train_rlct.py')
        original_path=self.root/'old-plan.json';save(original_path,original)
        argv=w.command(original,0,None);cp=w.checkpoint_path(argv,old_repo)
        loop=self.checkpoint(cp,progress(0,0),16,12);rng=loop.pop('runtime_rng')
        _,files=checkpoint_state(cp,progress(0,0),dict(segment_index=0,batch_offset=0,batch_count=16))
        updates=0;metrics=[]
        for step in range(1,17):
            skipped=step in (6,7,13,14);updates+=not skipped
            metrics.append(dict(step=step,**{'train/optimizer_step':updates,'train/skipped_empty_batch':int(skipped)}))
        data=self.root/'data.jsonl';data.write_text('fixture-data')
        audit=dict(job='6952064',scheduler_state='FAILED',checkpoint=str(cp),loop=loop,
            files={name:{k:record[k] for k in ('bytes','sha256')} for name,record in files.items()},
            runtime_rng_keys=list(rng),runtime_rng_sha256=hashlib.sha256(json.dumps(rng,sort_keys=True).encode()).hexdigest(),
            metrics=metrics,data_path=str(data),data_sha256=w.sha(data))
        audit_path=self.root/'audit.json';save(audit_path,audit)
        started=self.root/'started.json';save(started,dict(argv=argv,plan_sha256=w.sha(original_path),
                                                       gates=dict(source_commit='a'*40)))
        contract=self.root/'recovery.json';save(contract,dict(schema='rmct-first-child-recovery-contract-v2',
            destination_source_commit='b'*40,campaign_id=argv[argv.index('--run-name')+1].removesuffix('-s001'),
            origin_plan=identity(original_path),origin_started=identity(started),audit=identity(audit_path),checkpoint=str(cp)))
        self.plan['recovery_contract']=identity(contract);self.bind()
        return data,cp,contract

    def mocked_setting(self,data):
        setting=patch('ctm_data.adapters.mcq_bias.shared_qid_two_bias.SharedQidTwoBiasSetting')
        cls=setting.start();self.addCleanup(setting.stop)
        cls.return_value.data_path=data
        cls.return_value.load_datapoints.return_value=[dict(question_id=f'manifest-order-{i}') for i in range(32)]

    def test_explicit_recovery_preserves_12_and_next_unconsumed_slice(self):
        data,cp,contract=self.recovery_fixture();self.mocked_setting(data)
        with patch.object(w,'source_check'):
            recovered=w.recover_v2(self.plan,self.binding)
            self.assertEqual(w.verify_seal_v2(recovered,self.plan,self.binding),recovered)
        self.assertEqual(recovered['progress'],progress(16,12))
        self.assertEqual(recovered['consumed_question_ids'][0],'manifest-order-0')
        selection=next_slice(recovered['progress'],64)
        argv=w.command_v2(self.plan,recovered['progress'],selection,recovered)
        self.assertEqual(selection,dict(segment_index=1,batch_offset=0,batch_count=16))
        self.assertEqual(argv[argv.index('--resume-from')+1],'file://'+str(cp))

    def test_recovery_tamper_recipe_and_source_fail(self):
        data,cp,contract=self.recovery_fixture();self.mocked_setting(data)
        with patch.object(w,'source_check'):
            changed=copy.deepcopy(self.plan);a=changed['argv'];a[a.index('--max-new-tokens')+1]='1024'
            with self.assertRaises(ValueError):w.recover_v2(changed,self.binding)
            changed=copy.deepcopy(self.binding);changed['source_commit']='c'*40
            with self.assertRaises(ValueError):w.recover_v2(self.plan,changed)
            (cp/'optimizer.pt').write_bytes(b'tamper')
            with self.assertRaises(ValueError):w.recover_v2(self.plan,self.binding)

    def test_plan_changed_rejected(self):
        receipt=self.fresh_seal();self.plan['incorporated_commit']='c'*40
        with self.assertRaises(ValueError):w.verify_seal_v2(receipt,self.plan,self.binding)

    def test_actual_saved_audit_counter_regression(self):
        path=Path(__file__).resolve().parents[2]/'artifacts/rmct-tbsr-continuation-20260927/failure-6952064-audit.json'
        if not path.exists():self.skipTest('Local immutable failure audit not present')
        audit=json.loads(path.read_text());loop=audit['loop']
        self.assertEqual((loop['global_step'],loop['optimizer_step']),(16,12))
        metrics=[r for r in audit['metrics'] if 'train/optimizer_step' in r]
        self.assertEqual(len(metrics),16)
        self.assertEqual([r['step'] for r in metrics if r['train/skipped_empty_batch']],[6,7,13,14])
        self.assertEqual(next_slice(progress(loop['global_step'],loop['optimizer_step']),64),
                         dict(segment_index=1,batch_offset=0,batch_count=16))

    def test_loader_ignoring_slice_is_rejected(self):
        parent=dict(checkpoint='/saved',progress=progress(68,62))
        selected=next_slice(parent['progress'],64)
        argv=w.command_v2(self.plan,parent['progress'],selected,parent)
        with patch('ctm_data.adapters.mcq_bias.shared_qid_two_bias.SharedQidTwoBiasSetting') as setting:
            setting.return_value.load_datapoints.return_value=[dict(question_id=str(i)) for i in range(32)]
            with self.assertRaises(ValueError):w.verify_data_slice(argv,selected)

    def test_no_progress_pause_never_invokes_training(self):
        receipt=dict(progress=progress(500,0,no_progress_batches=500))
        with patch.object(w,'verify_seal_v2',return_value=receipt),patch.object(w.subprocess,'run') as launch:
            with self.assertRaisesRegex(ValueError,'not convergence'):
                w.train_to_boundary(self.plan,self.binding,self.root/'campaign',64,{},receipt,{})
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
