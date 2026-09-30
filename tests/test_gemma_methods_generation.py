import unittest
from experiments.gemma4_methods.evaluate import generation_for, CAPS


class GenerationPolicyTests(unittest.TestCase):
    def test_alignment_path_explicitly_enables_native_thinking(self):
        from experiments.gemma4_methods.train import alignment_processor
        from unittest.mock import Mock
        official=Mock()
        official.apply_chat_template.return_value='native'
        wrapped=alignment_processor(official)
        wrapped.apply_chat_template([{'role':'user','content':'Question'}],tokenize=False,
                                    add_generation_prompt=True)
        self.assertIs(official.apply_chat_template.call_args.kwargs['enable_thinking'],True)
        with self.assertRaises(ValueError):
            wrapped.apply_chat_template([],enable_thinking=False)
    def test_more_workers_preserve_exact_frozen_population(self):
        from experiments.gemma4_methods.evaluate import shard_question_ids,WORKER_CHOICES
        qids=[str(i) for i in range(50)]
        for workers in WORKER_CHOICES:
            shards=[shard_question_ids(qids,rank,workers) for rank in range(workers)]
            flattened=[qid for shard in shards for qid in shard]
            self.assertEqual(len(flattened),50)
            self.assertEqual(set(flattened),set(qids))
            self.assertTrue(all(shards))
        with self.assertRaises(ValueError):
            shard_question_ids(qids,16,16)
    def test_actual_updates_and_attempts_remain_distinct(self):
        from experiments.gemma4_methods.progress import record_update, bounded_end
        s = record_update({'step': 12, 'pending': []}, attempt=16, loss=.2, question_ids=['a','b'])
        self.assertEqual((s['step'], s['attempts']), (13, 17))
        self.assertEqual(bounded_end(60, 16), 64)
        with self.assertRaises(RuntimeError):
            bounded_end(64, 16)
    def test_exact_caps_and_thinking(self):
        for dataset, cap in [('logiqa', 20480), ('hellaswag', 20480), ('hle-text-mc', 65536)]:
            g = generation_for(dataset, bad_words=['reserved'])
            self.assertEqual(g['max_tokens'], cap)
            self.assertIs(g['extra_body']['chat_template_kwargs']['enable_thinking'], True)
            self.assertEqual(g['extra_body']['bad_words'], ['reserved'])

    def test_unknown_dataset_fails_closed(self):
        with self.assertRaises(ValueError):
            generation_for('LogiQA', bad_words=[])

    def test_configs_are_not_shared_mutable_objects(self):
        g = generation_for('logiqa', bad_words=[])
        g['extra_body']['chat_template_kwargs']['enable_thinking'] = False
        self.assertTrue(generation_for('logiqa', bad_words=[])['extra_body']['chat_template_kwargs']['enable_thinking'])
