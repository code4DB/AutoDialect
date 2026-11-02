"""SQL UT split, focused group execution, and validation helpers.

1. sql_ut_splitter.py exposes the main-process split-manifest API.
2. sql_ut_splitter_worker.py isolates SQLGlot parsing and raw split generation
   in a subprocess.
3. group_execution.py plans focused rule-group queues and batches.
4. split_validator.py validates focused units, variants, and parent SQL.
5. sql_ut_workset.py selects prompt-sized evidence from completed manifests.
"""
