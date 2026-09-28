# Imports must survive stale bytecode

Streamlit crashed with `ImportError: INSTRUMENT_PRESETS` though the name existed — mixed 3.12/3.13 `__pycache__` served a stale module. Fix: delete `__pycache__`, move imports to file top, keep a local fallback dict. Lesson: raw tracebacks must never reach a layman user.
