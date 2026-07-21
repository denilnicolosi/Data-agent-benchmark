# Prompts for LLM Code Generators
#
# Placeholders use Langfuse's native {{mustache}} syntax, compiled by
# prompt_store (Langfuse prompt.compile() when the text comes from Langfuse,
# an equivalent local substitution otherwise), so the prompts are 100%
# Langfuse-compatible: variables are recognized in the Playground and UI
# experiments. Single literal braces remain safe (mustache only reacts to
# {{...}}). The analytical request is deliberately the ONLY specification of
# what to compute: don't hardcode any concrete analysis in these templates.

BASE_INSTRUCTIONS = """
You are generating Python code for an automated benchmark pipeline.

Write a Python script whose entry point is a function `analyze(db_path: str)` that answers the analytical request below by processing transaction data from a SQLite database.

Database Schema:
The database at `db_path` has a table named `transactions` with the following columns:
- `transaction_id` (INTEGER, Primary Key)
- `user_id` (INTEGER)
- `amount` (REAL)
- `timestamp` (TEXT, format 'YYYY-MM-DD HH:MM:SS')
- `category` (TEXT)

Execution environment (IMPORTANT: target these exact versions, NOT older APIs):
Python 3.12 with pandas 3.x, polars 1.x, numpy 2.x, dask 2026.x, pyarrow, and the stdlib `sqlite3`. Nothing else is installed (no connectorx, no sqlalchemy). Do not use functions, parameters or aliases that were deprecated/removed in these versions.

Analytical Request (the ONLY specification of what to compute):
"{{analytical_request}}"

How to fill the structured output fields:
- `reasoning`: 2-4 plain-text sentences explaining your implementation approach, the library-specific optimizations you applied, and any assumptions you made about the request.
- `code`: the complete runnable Python script as raw source code. No markdown fences, no prose, nothing outside the script itself. This field must contain everything needed to run.

Rules for the script in `code`:
1. It MUST define exactly one entry-point function `def analyze(db_path: str):`. Helper functions are allowed; no module-level statements besides imports.
2. Do NOT generate mock data: connect with `sqlite3` and read the `transactions` table from `db_path`.
3. `analyze` MUST return a dictionary mapping short snake_case keys to the results that answer the request (DataFrames, arrays, dicts or scalars). Return aggregated / preview-sized results, never the full raw table.
4. Do not print, plot, write files, spawn processes, or access the network; just compute and return.
5. The script is executed unmodified in a sandboxed benchmark container and profiled: optimize for speed and memory.
"""

PANDAS_PROMPT = BASE_INSTRUCTIONS + """
Specific Library: PANDAS
Implement the analysis with pandas: load the data via `pd.read_sql_query` on a `sqlite3` connection, convert `timestamp` with `pd.to_datetime`, and use vectorized pandas operations (groupby, rolling, merge); no Python-level row loops.
pandas 3.x notes: copy-on-write is always on (chained assignment never works, assign via `.loc` or reassign the column); `DataFrame.append`/`Series.append` no longer exist (use `pd.concat`); `inplace=` is gone from most methods.
"""

DASK_PROMPT = BASE_INSTRUCTIONS + """
Specific Library: DASK
Implement the analysis with Dask DataFrames (`dask.dataframe`): read via pandas and convert with `dd.from_pandas(..., npartitions=4)` (or more) to simulate parallel out-of-core execution. Note that rolling/window operations in dask require sorting and setting the index first, or a per-group `map_partitions` approach. Call `.compute()` before returning, so the dictionary contains concrete pandas objects.
Dask 2026.x notes: the query-planning (expressions) DataFrame is the default and the underlying pandas is 3.x (see the pandas notes on removed APIs); when using `map_partitions`/`apply`, pass an explicit `meta=` to avoid inference warnings.
"""

NUMPY_PROMPT = BASE_INSTRUCTIONS + """
Specific Library: NUMPY
Implement the analysis with NumPy vectorization: fetch the rows via sqlite3, convert the relevant columns to numpy arrays, and use vectorized tools (np.unique, np.bincount, np.add.at, np.convolve, argsort) instead of Python loops. You may convert the final numpy results to pandas DataFrames for readability.
NumPy 2.x notes: the old aliases are gone, use `np.nan` (not `np.NaN`), `np.float64`/`np.int64` (not `np.float_`/`np.int_`), and `np.inf` (not `np.Inf`).
"""

POLARS_PROMPT = BASE_INSTRUCTIONS + """
Specific Library: POLARS
Implement the analysis with Polars, preferring lazy evaluation (LazyFrame + `collect()`): parse timestamps to Datetime and use Polars expressions (group_by, rolling/window functions) for the computation. Return Polars DataFrames, or convert with `.to_pandas()`.
Polars 1.x data loading (this is where old code breaks), follow exactly:
- open a stdlib connection and pass the OBJECT: `conn = sqlite3.connect(db_path)` then `pl.read_database("SELECT ...", connection=conn)`;
- the `connection_uri` parameter does NOT exist anymore, and `pl.read_database_uri` is unusable here (connectorx is not installed);
- alternatively fetch rows with sqlite3 and build `pl.DataFrame` manually.
Method renames since old versions: `group_by` (not `groupby`), `pl.col(...).cast(...)`, `.str.to_datetime(...)` for parsing timestamp strings.
"""

# Report commentary, NOT a code-generation prompt. Variables ({{mustache}}):
# analytical_request, selected, context (per-library blocks: status, metrics,
# result preview, and the exact code that ran).
REPORT_COMMENTARY_PROMPT = """
You are reviewing the outcome of a benchmark session that compared 4 Python data-processing implementations (pandas, dask, numpy, polars) of the same analytical request. Each approved script ran in an identical isolated container (no network, same CPU/RAM caps) and executions were serialized, so the measurements are directly comparable.

Analytical request:
"{{analytical_request}}"

The human reviewer selected {{selected}} as the winning strategy.

Below, for each library: final status, execution metrics (elapsed seconds, net peak RSS), a preview of the returned result, and the exact code that ran.

{{context}}

Write a concise commentary in Markdown (start directly with prose or ### subsections, no top-level title), covering:
- WHY the fastest and slowest implementations performed the way they did, pointing at concrete choices in their code (eager vs lazy, SQL pushdown vs full-table load, vectorization, partitioning, unnecessary copies);
- whether the returned results agree across libraries - flag any discrepancy visible in the previews;
- whether the human's selection is reasonable given the numbers, and in which scenario a different branch would be the better choice;
- for failed or rejected branches, what the failure/rejection cost in terms of missing evidence.
Under 350 words. Be specific to THIS session's data and code, no generic library trivia.
"""
