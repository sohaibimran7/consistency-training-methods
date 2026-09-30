import unittest
from experiments.rmct_restart_20260928.qwen_progress import progress,next_slice,advance,validate_loop


class ProgressTests(unittest.TestCase):
    def test_saved_16_batches_12_updates_is_not_relabelled(self):
        state=progress(16,12)
        validate_loop(dict(step=16,global_step=16,optimizer_step=12,final=True,accumulated_grads=0),state)
        self.assertEqual(next_slice(state,64),dict(segment_index=1,batch_offset=0,batch_count=16))
        with self.assertRaises(ValueError):
            validate_loop(dict(step=16,global_step=16,optimizer_step=16,final=True,accumulated_grads=0),state)

    def test_exact64_with_skips_never_overshoots(self):
        state=progress(64,60)
        selection=next_slice(state,64)
        self.assertEqual(selection['batch_count'],4)
        state=advance(state,selection,62)
        self.assertEqual(next_slice(state,64),dict(segment_index=4,batch_offset=4,batch_count=2))
        state=advance(state,next_slice(state,64),64)
        self.assertIsNone(next_slice(state,64))
        self.assertEqual(state['sampled_batches'],70)

    def test_zero_signal_advances_questions_not_optimizer(self):
        state=progress(16,12)
        after=advance(state,next_slice(state,64),12)
        self.assertEqual(after,progress(32,12,no_progress_batches=16))

    def test_no_progress_is_bounded_not_convergence(self):
        state=progress(496,0,no_progress_batches=496)
        selection=next_slice(state,64)
        self.assertEqual(selection['batch_count'],4)
        state=advance(state,selection,0)
        with self.assertRaises(ValueError):next_slice(state,64)

    def test_malformed_and_overshoot_rejected(self):
        for b,o in [(12,16),(True,1),(-1,0)]:
            with self.assertRaises(ValueError):progress(b,o)
        with self.assertRaises(ValueError):next_slice(progress(80,65),64)
        with self.assertRaises(ValueError):next_slice(dict(progress(16,12),batch_offset=1),64)
        with self.assertRaises(ValueError):advance(progress(16,12),dict(segment_index=0,batch_offset=0,batch_count=16),13)

    def test_every_reachable_chunk_bounds_update_count(self):
        for batches in range(65):
            for updates in range(min(batches,63)+1):
                state=progress(batches,updates)
                selection=next_slice(state,64)
                for delta in range(selection['batch_count']+1):
                    after=advance(state,selection,updates+delta)
                    self.assertLessEqual(after['optimizer_updates'],64)
                    self.assertEqual(after['sampled_batches'],batches+selection['batch_count'])


if __name__=='__main__':unittest.main()
