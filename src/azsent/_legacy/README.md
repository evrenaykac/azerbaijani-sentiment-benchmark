# Retired modules

`data_prep.py`, `gold.py`, `splits.py` — the three stages that, in v1,
recomputed deduplication, gold assembly and grouped splitting **on every run**.
In v2 this work moved to data-preparation time (`tools/repack_dataset.py`), and
only verification remains on the run side (`azsent/ingest.py`).

Why they were retired: all three sat on the critical path of every run, and
each could fail silently — gold parts filled from another source, domain names
that did not match, quarantine logic applied only in part. The split now lives
in the file itself and can be audited; the pipeline does not produce it, it
tests it.

They are kept here for reference and are not imported.
