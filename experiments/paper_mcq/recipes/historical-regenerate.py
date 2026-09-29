"""Rerun the parser-fixed figure recipes on the 64k-recovery-merged samples."""
import json, os, shutil, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
FIXED = R / 'artifacts/parser-fixed-20260923'
OUT = R / 'artifacts/parser-fixed-64k-20260925'
STAGE = OUT / 'figures-final'; STAGE.mkdir(exist_ok=True)
recipes = {
 'paper-behavioural-plots-20260917': ['build.py'],
 'paper-behavioural-tbsr-20260917': ['regenerate.py'],
 'joint-verbalisation-seen-heldout-20260917': ['build.py'],
 'answer-transitions-seen-heldout-20260917': ['build.py'],
 'accuracy-invariance-tradeoff-20260917': ['build.py'],
 'biased-accuracy-20260918': ['build.py'],
 'accuracy-by-bias-paper-20260918': ['build.py'],
 'away-from-bias-switch-20260918': ['build.py'],
 'conditional-verbalisation-20260918': ['split.py', 'paper.py'],
 'switch-direction-composition-20260918': ['build.py', 'composite.py'],
 'parse-grade-coverage-20260917': ['build.py'],
 'towards-bias-switch-standard': ['build.py'],
 'bias-verbalisation-standard': ['build.py'],
}
OLD_BANNER = 'Corrected MCQ answers; Base clean-dependent values remain PROVISIONAL (raw clean responses unavailable).'
NEW_BANNER = ('Corrected MCQ answers + selective 64k reruns + RMCT IID top-up (100 QIDs/dataset). 12 BCT reruns missing. '
              'Base clean-dependent values PROVISIONAL.')
for folder, files in recipes.items():
    dest = STAGE / folder; dest.mkdir(exist_ok=True)
    for f in files:
        text = (FIXED / 'figures-final' / folder / f).read_text()
        assert OLD_BANNER in text, (folder, f)
        text = text.replace(OLD_BANNER, NEW_BANNER)
        if folder == 'parse-grade-coverage-20260917':
            # The historical grading audit predates the RMCT top-up; every cell now has 100 prompts.
            old = "assert len(subset)==record['generated']"; assert old in text
            text = text.replace(old, "assert len(subset)==100")
            old = "assert totals['generated']==12000 and totals['biased_unparsed']+totals['parsed_graded']+totals['parsed_missing_grade']==12000"
            assert old in text
            text = text.replace(old, "assert totals['generated']==12600 and totals['biased_unparsed']+totals['parsed_graded']+totals['parsed_missing_grade']==12600")
        (dest / f).write_text(text)
sources = json.loads((FIXED / 'sources.json').read_text())
sources.update(json.loads((OUT / 'recovery-sources.json').read_text()))
(OUT / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
pb = STAGE / 'paper-behavioural-plots-20260917'
shutil.copy2(OUT / 'samples.json', pb / 'samples.json'); shutil.copy2(OUT / 'sources.json', pb / 'sources.json')
template = 'methods-verbalisation-all-seven-20260916/bias_acknowledged-vs-base/chart-spec.json'
(STAGE / template).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(FIXED / 'figures-final' / template, STAGE / template)
env = dict(os.environ, PYTHONPATH=str(R), CTM_CONDITIONAL_PERMUTATIONS='108000', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/ctm-monitor-matplotlib')
status = {}
only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
for folder, files in recipes.items():
    if only and folder != only: continue
    for f in files:
        key = f'{folder}/{f}'; print('START', key, flush=True)
        with (STAGE / folder / (f + '.log')).open('w') as log:
            p = subprocess.run([sys.executable, str(STAGE / folder / f)], env=env, stdout=log, stderr=subprocess.STDOUT)
        status[key] = p.returncode
        prior = json.loads((OUT / 'figure-build-status.json').read_text()) if (OUT / 'figure-build-status.json').exists() else {}
        prior.update(status); (OUT / 'figure-build-status.json').write_text(json.dumps(prior, indent=2) + '\n')
        print('END', key, p.returncode, flush=True)
        if folder == 'paper-behavioural-plots-20260917' and p.returncode: sys.exit(1)
