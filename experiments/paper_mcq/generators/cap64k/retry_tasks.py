import json
from pathlib import Path
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

def tasks(shard):
    data=json.loads(Path(shard).read_text())
    return [Task(name='cap64k-'+data['method'],dataset=[Sample(**s) for s in data['samples']],
                 solver=generate(),scorer=None)]
