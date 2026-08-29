# Contributing

Use focused commits and keep input data, generated results, and paper figures in
their existing directories. New experiments should record the random seed, package
versions, runtime, machine/backend, input filenames, and a checksum or immutable
identifier for external data.

Before opening a change:

```bash
python3 -m compileall -q scripts
python3 scripts/verify_paper_claims.py
```

Never commit IBM credentials or local environment files. Hardware submission must
remain opt-in; a script should not contact or bill a remote service by default.
