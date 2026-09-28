"""Fail-closed Gemma thinking-mode probe; no generation or token cap involved."""
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping


def require_run_approval(root, *, scope, cap, approval_path=None):
    """New thinking-on runs must not inherit legacy token-cap approval."""
    path = Path(approval_path) if approval_path else Path(root) / 'thinking-run-approval.json'
    if not path.exists():
        raise RuntimeError('New thinking-enabled run needs explicit scoped cap approval: ' + str(path))
    approval = json.loads(path.read_text())
    expected = {'run_root': str(Path(root).resolve()), 'scope': scope,
                'enable_thinking': True, 'generated_token_cap_including_reasoning': cap}
    if any(approval.get(k) != v for k, v in expected.items()) or not approval.get('user_approval_reference'):
        raise ValueError('Thinking-run approval scope mismatch')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def attest_thinking(processor):
    messages = [{'role': 'user', 'content': 'What is 2 + 2?'}]
    ids = {}
    for mode in (True, False):
        kwargs = dict(add_generation_prompt=True, enable_thinking=mode)
        text = processor.apply_chat_template(messages, tokenize=False, **kwargs)
        tokens = processor.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        if isinstance(tokens, Mapping):
            tokens = tokens['input_ids']
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], list):
            tokens = tokens[0]
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError('Expected flat integer prompt tokens')
        encoded = processor.encode(text, add_special_tokens=False)
        if list(encoded) != tokens:
            raise ValueError('Rendered text and native template tokenization disagree')
        if mode:
            if '<|turn>system\n<|think|>\n' not in text or not text.endswith('<|turn>model\n'):
                raise ValueError('Thinking-enabled Gemma prompt signature missing')
        elif not text.endswith('<|turn>model\n<|channel>thought\n<channel|>'):
            raise ValueError('Thinking-disabled control signature missing')
        ids[str(mode).lower()] = tokens
    if ids['true'] == ids['false']:
        raise ValueError('Thinking flag did not change prompt tokens')
    template = processor.chat_template
    return {'enable_thinking': True, 'probe_messages': messages, 'prompt_token_ids': ids,
            'template_sha256': hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest(),
            'probe_sha256': hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()}
