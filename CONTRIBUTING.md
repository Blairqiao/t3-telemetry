# Contributing

Thanks for helping. Bug reports, harness parsers and price corrections are all welcome.

## Development setup

```bash
git clone https://github.com/Blairqiao/t3-telemetry && cd t3-telemetry
uv run pytest          # the test suite
uv run t3-top          # run against your own data
```

The tests use synthetic fixtures only. `tests/conftest.py` points `HOME` and every path variable at a temporary directory, so a test run never reads your real `~/.claude`, `~/.codex` or `~/.t3`. Keep it that way: new tests should build their own files under `tmp_path`.

Never commit real session files, T3 databases or a telemetry store. They contain your prompts.

## How the pieces fit

1. `collector.py` finds session files for each harness and passes new bytes to a parser in `sources/`.
2. Each parser returns a `ParseResult`: sessions, steps (one per model call, with token counts), turns and activity items. The collector writes them to the store (`store.py`).
3. `t3_mirror.py` copies T3's thread tree and its session bindings into the store, so a native session can be shown under the T3 thread that started it.
4. `usage.py` and `thread_view.py` read the store. `ui.py` and `cli.py` render it.

## Adding a harness

T3 can drive harnesses that `t3-top` has no parser for, OpenCode for example. Their T3 threads still count toward usage, from the per-turn totals T3 records, under **Other** (see `t3_mirror._fallback_session`). A parser gives them their own provider label, per-call steps, activity detail, and coverage of sessions run outside T3.

Use `src/t3_telemetry/sources/codex.py` as the model. It's a self-contained JSONL parser that supports incremental reads.

### 1. Write the parser: `sources/<name>.py`

Return a `ParseResult` from `t3_telemetry.store`:

- **One `SessionRecord` per native session.** `session_key` is `"<provider>:<native id>"`, where the native id is the same id T3 stores as the provider thread's `native_id`. That shared id is how T3 threads find their sessions. Set `origin_hint` to `cli`, `sdk` or `unknown` when the file says how the session was started.
- **One `StepRecord` per model call.** `event_id` must be stable and unique within the session, so that re-parsing a file doesn't double-count. Timestamps go through `utc_iso`.
- **`offset`.** The byte offset you've parsed up to. The collector passes it back next time, so only appended bytes are read. Stop at the last complete record. `sources/_jsonl.read_jsonl` handles this for JSONL files.
- **Turns and items** are optional. They feed the Activity tab. Bound any text with `bounded(text, MAX_TEXT_CHARS)` or `MAX_ITEM_CHARS`, so the store never holds full transcripts.

**Token convention.** Every harness must report tokens the same way, or totals and costs will be wrong:

| Field | Meaning |
|---|---|
| `input_tokens` | Uncached input only. If the harness reports total input, subtract cache reads and cache writes. |
| `cache_read_tokens` | Input served from cache. |
| `cache_creation_tokens` | Input written to cache. |
| `output_tokens` | Everything the model generated, **including** thinking. |
| `thinking_tokens` | The thinking/reasoning part of `output_tokens`. Don't add it to the total again. |

Total = `input_tokens + cache_read_tokens + cache_creation_tokens + output_tokens`.

### 2. Register it

There are five touch points:

1. **`paths.py`**: add a function for the harness's home directory with an environment-variable override, following `codex_home()`.
2. **`collector.py`**: add a field to `SourceRoots`, fill it in `default_roots`, then add the glob and the parser call in `Collector.sync`.
3. **`providers.py`**: add the key to `PROVIDER_KEYS` and `FAMILY_LABELS`, and teach `provider_key` to recognise the T3 driver name. Once it does, T3 threads bind to your parsed sessions instead of the "Other" fallback.
4. **`usage.py`**: add the label to `PROVIDER_ORDER`, before `"Other"`.
5. **`ui.py`**: add a colour to `PROVIDER_STYLES` and a bar glyph to `PROVIDER_GLYPHS`.

Also add the harness's default context window to `context_limits.py` and its models to `prices.py` if they're missing.

### 3. Test it

Add `tests/test_source_<name>.py` with small hand-written session files, following `tests/test_source_codex.py`. Cover at least:

- the token mapping, using a record that has cache reads and thinking;
- an incremental read: parse, append a record, parse from the returned offset, and check that nothing is counted twice;
- a truncated final line, which must be left for the next read.

Then update the "Supported harnesses" and "Where data comes from" tables in the README.

## Updating prices

Rates live in `src/t3_telemetry/prices.py`, in US dollars per million tokens.

- `_TABLE` maps a model key to a list of `(effective_date, Rate(input, output, cache_read, cache_write))`. Use `None` for the first entry and a `date` for later price changes. Model names match on prefix, and a trailing `-YYYYMMDD` is ignored.
- `_LONG` holds long-context rates: `(prompt_token_threshold, Rate)`.
- When you change rates, update `PRICES_CHECKED` and the comment at the top of the file, and list your sources. Mark any third-party source with `# third-party, verify`.

Add a case to `tests/test_prices.py` for any new model.

## Pull requests

- Run `uv run pytest` before opening a PR.
- Keep T3 and harness files read-only. `t3-top` must never write to anything it doesn't own.
- Describe user-visible changes in the README.
