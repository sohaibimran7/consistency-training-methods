"""Terminal-first MCQ extraction; completion status must be gated by caller.

Syntactic extraction is not semantic grading. Unresolved outputs remain missing.
Frozen pilot implementation promoted for integrated review, not GPU clearance.
"""
import re
from ctm_data.adapters.mcq_bias.answer_parsing import parse_indicator_answer

def parse_terminal_first(text,allowed,break_words=()):
    # Completion status is a required caller gate, not inferable from text.
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    elif '<think>' in text:return None
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    if not lines:return None
    line=lines[-1]
    line=re.sub(r'\*\*([^*]+)\*\*',r'\1',line)
    line=re.sub(r'\\(?:boxed|textbf|text)\s*\{','',line)
    terminal=re.fullmatch(r'(?:(?:the|my)\s+)?(?:(?:best|correct|final)\s+)?answer\s*(?:is\s*:?|:)\s*[($\s]*([A-Z])[\s).}$]*',line,re.I)
    if terminal and text.count('```')%2==0:
        label=terminal.group(1).upper()
        return label if label in allowed else None
    return parse_indicator_answer(text,allowed,break_words)
