# Evaluation protocol

- Never use any output-token, generation-token, completion-token, or `max_tokens`/`max_new_tokens` cap in a run unless the user has explicitly approved that exact cap for that specific run. This includes caps inherited from defaults, papers, upstream implementations, saved configurations, or prior runs. Before requesting approval, state the exact proposed cap, why it is needed, whether the model emits reasoning tokens, and the expected truncation risk. Absence of an explicit per-run approval means the run must not use a token cap.
