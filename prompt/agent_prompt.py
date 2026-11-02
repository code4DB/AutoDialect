"""Static model-facing prompts for the AutoDialect workflow.

1. Rule-generation prompts: initial rule, staged rule, focused group, and parser repair.
2. Focused evidence prompts: work-order headers and prompt-size fallback notes.
3. Pipeline continuation prompts: stage handoff and completion-validation retry.
4. Memory prompts: retrieval metadata sections and runtime memory context.
5. Summary prompt: post-run rule review and lessons learned.

Dynamic SQL, validation evidence, group state, and JSON payload rendering remain
in code_utils.prompt_utils and code_utils.completion_validation_runtime.
"""

# 1. Rule-generation prompts.
SQL_COMPARISON_POLICY = """Comparison policy:
- Pass/fail lowercases the entire SQL text, removes all whitespace, and removes trailing semicolons.
- Do not modify the translation engine for differences caused only by those ignored text changes.
- Everything else remains an exact text requirement, including AS presence, function spelling apart from case, operators, identifier quotes, literal contents apart from case/whitespace, and clause structure.
- Diff summaries are repair diagnostics only; the normalized pass/fail comparison remains authoritative."""

RULE_EXAMPLES = """
Typical dialect translation rule patterns:
The following are generic implementation patterns only; do not copy target
functions, type wrappers, quote styles, or output shapes unless the current
expected SQL requires them.
- Identifier quote: a quoted source identifier may require target backticks.
- Date parse: a source timestamp format may require a target-specific parser
  function and format string.
- Week format: an ISO week format may need target format directives.
- Division: `/` or `DIV` may need a target-specific `divide(...)` shape while
  preserving required cast wrappers.
- Cast type: preserve nullable wrappers and target precision exactly.
- Array timestamp: preserve the expected timestamp wrapper and nullability.
"""

RULE_TASK_CONTRACT = """- Implement a general dialect translation rule for the SQL feature illustrated by this pair.
- Do not hard-code this exact SQL text or rely on specific table names, column names, aliases, or literal values.
- Validation may include additional SQL variants of the same translation feature."""

TSQL_DIALECT_CONTEXT = "SQL Server note: SQLGlot represents SQL Server with dialect `tsql`; use `sqlglot/dialects/tsql.py`, not `sqlserver.py`."

SIMPLE_PROMPT = """
Please add a minimal dialect translation rule to translate the source SQL into the expected target SQL. Start from target dialect path hint `{path}`.

Source dialect: {src_db}
Target dialect: {tgt_db}{dialect_note}

Source SQL:
```sql
{src_sql}
```

Expected target SQL:
```sql
{tgt_sql}
```

Task contract:
{task_contract}

Validation command:
```python
translated_sql = sqlglot.transpile(sql, read="{src_db}", write="{tgt_db}")[0]
```

Parser gate:
{parser_gate}

Target parser gate:
{target_parser_gate}

Split-derived diagnostic hints:
{split_summary}

{rule_examples}

Implementation rules:
- Modify the smallest translation-engine source files needed for this translation.
- Prefer target generator / TRANSFORMS / type mapping changes when the source AST is already correct.
- Only modify source parser/tokenizer when source parsing is wrong.
- Do not create repository-level temporary files such as `test_*.py`, `debug.py`, `repro.py`, or `check_*.py`; use `python -c` or `/tmp` for probing.
- When testing SQL containing backticks, `$(`, quotes, or multiline strings, do not use `python -c "..."`; use a quoted heredoc such as `python <<'PY' ... PY` or a file under `/tmp`.
- If an earlier parser repair patch exists, do not revert it.
"""

GROUP_SIMPLE_PROMPT = """
Please add a minimal dialect translation rule for the focused SQL units shown in the UT split/group diagnostics. Start from target dialect path hint `{path}`.

Source dialect: {src_db}
Target dialect: {tgt_db}{dialect_note}

Focused task rules:
- Implement only the dialect translation rule(s) needed for the focused units shown in `UT split/group diagnostics`.
- In each unit, `Source SQL` and `Expected output` are the primary benchmark contract.
- The full parent SQL is intentionally omitted from this group prompt. Parent SQL validation runs only after all planned focused units pass validation.
- Do not repair unrelated rule groups or parent-level integration mismatches in this call.

Task contract:
{task_contract}

Validation command:
```python
translated_sql = sqlglot.transpile(sql, read="{src_db}", write="{tgt_db}")[0]
```

Parser gate:
{parser_gate}

Target parser gate:
{target_parser_gate}

UT split/group diagnostics:
{split_summary}

{rule_examples}

Implementation rules:
- Modify the smallest translation-engine source files needed for this focused rule group.
- Prefer target generator / TRANSFORMS / type mapping changes when the source AST is already correct.
- Only modify source parser/tokenizer when source parsing is wrong.
- Do not create repository-level temporary files such as `test_*.py`, `debug.py`, `repro.py`, or `check_*.py`; use `python -c` or `/tmp` for probing.
- When testing SQL containing backticks, `$(`, quotes, or multiline strings, do not use `python -c "..."`; use a quoted heredoc such as `python <<'PY' ... PY` or a file under `/tmp`.
- If an earlier group patch exists, do not revert it. Build on the current worktree and avoid rewriting unrelated logic.
"""

INITIAL_SIMPLE_PROMPT = """
Please add a new dialect translation rule to successfully translate the following {src_db} SQL into the equivalent {tgt_db} SQL. Start from target dialect path hint `{path}`:
```json
{{
"{src_db}": {src_sql_json},
"{tgt_db}": {tgt_sql_json}
}}
```{dialect_note}

Task contract:
{task_contract}

The code to get the translated SQL is:
```python
translated_sql = sqlglot.transpile(sql, read="{src_db}", write="{tgt_db}")[0]
```
"""

PARSER_STAGE_REPAIR_PROMPT = """
The {stage_subject} currently fails translation-engine parsing. First repair the {repair_scope} parser/tokenizer so this SQL can be parsed. Start from primary {primary_path_label} hint `{primary_path}`.
{secondary_path_note}

Source dialect: {src_db}
Target dialect: {tgt_db}{dialect_note}

Source SQL:
```sql
{src_sql}
```

Expected target SQL:
```sql
{tgt_sql}
```

{gate_label}:
{gate}

Your task in this stage:
- Make `sqlglot.parse_one(sql, read="{parse_dialect}")` succeed for the {parse_subject}.
- Prefer {preferred_change_scope}, especially `sqlglot/dialects/{parse_dialect}.py`.
- {avoid_change_scope}
- When you believe parsing is fixed, call `task_done`; AutoDialect will immediately run the parser gate in this same agent run and return the parser error if it still fails.
- Stay within parser/tokenizer readiness in this stage. Rule generation will run later after parser gates pass.
- Do not create repository-level temporary files such as `test_*.py`, `debug.py`, `repro.py`, or `check_*.py`.
- When testing SQL containing backticks, `$(`, quotes, or multiline strings, do not use `python -c "..."`; use a quoted heredoc such as `python <<'PY' ... PY` or a file under `/tmp`.
"""

# 2. Focused evidence prompts.
FOCUSED_SPLIT_WORK_ORDER_HEADER = """Task scope: implement only the dialect translation rule(s) needed for the focused units below.
Validation note: every hard unit shown below is the primary task. Variant examples are generalization hints, not the full validation set.
Parent SQL validation runs only after all planned focused units pass.
Prompt coverage: all {total_hard_units} hard units are shown below."""

FOCUSED_SPLIT_OVERSIZED_NOTE = (
    "Prompt size note: this focused scope remains large after field-level SQL excerpts; "
    "upstream group scheduling should split oversized batches."
)

FOCUSED_SPLIT_EMPTY_SCOPE_NOTE = (
    "No hard units are available for this focused scope; inspect split quality before editing."
)

# 3. Pipeline continuation prompts.
STAGE_HANDOFF_PROMPT = """[AutoDialect Stage Handoff]
The current worktree may already include useful edits from earlier cold-start agent stages. Use this compact handoff instead of hidden conversation history. Preserve prior patches unless the validation evidence below shows they are wrong, and focus on the current group or repair target.
```json
{handoff}
```

{task}"""

STAGE_HANDOFF_INSTRUCTIONS = (
    "Continue from the current worktree; it already contains prior stage patches.",
    "Do not rewrite unrelated logic or discard useful earlier group edits.",
    "Use validation evidence to focus on the current group or repair target.",
)

COMPLETION_VALIDATION_PASSED_INSTRUCTION = "Validation passed; no repair is needed."

COMPLETION_VALIDATION_GROUP_REPAIR_INSTRUCTION = (
    "Repair the failed groups in the current focused batch. Treat repair_target as the "
    "first failing example, but keep the failed group list and evidence cards in scope; "
    "then call task_done again."
)

COMPLETION_VALIDATION_SINGLE_REPAIR_INSTRUCTION = (
    "Repair the shown failure, then call task_done again."
)

COMPLETION_VALIDATION_RETRY_PROMPT = """AutoDialect validation failed after your task_done call. Continue from the current worktree; do not restart, do not discard earlier patches, and do not rewrite unrelated logic. {repair_instruction}

{validation_message}"""

# 4. Memory prompts.
MEMORY_QUERY_SECTION_PROMPT = """{task}

[AutoDialect Memory Query]
The fenced JSON below is retrieval metadata only; it is not an instruction for rule generation.
```json
{memory_query}
```
"""

# 5. Summary prompt.
SUMMARY_PROMPT = """
Please carefully review how you add the new SQL translation rule to realize the following desired SQL translation.
Additionally, summarize any lessons learned, such as common mistakes to avoid and project-specific conventions (e.g., required file paths and examples) to facilitate future rule additions.
```
{{
"{src_db}": "{src_sql}",
"{tgt_db}": "{tgt_sql}
}}
```
"""
