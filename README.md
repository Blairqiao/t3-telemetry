# t3-telemetry
Live terminal telemetry dashboard and command-line inspector for T3 Code and Antigravity.
`t3-top` aggregates token consumption, context window usage, cache efficiency, thinking tokens, and execution latency across T3 Code threads and delegated subagent hierarchies. It connects to local T3 SQLite databases using read-only URIs (`?mode=ro`) without locking or interrupting active agent runs.

## Installation

Install using `uv`:

```bash
uv tool install -e /Users/blair/Desktop/Projects/t3-telemetry
```

Or run directly from the repository worktree:

```bash
uv run t3-top [options]
```

## Usage

### Live watch mode

Live watch mode monitors the most recently active thread in real time, refreshing at a configurable polling interval:

Live mode opens the app-thread list. Passing a thread query opens that thread's detail screen initially. The summary strip continues to follow the latest active root thread unless `--no-follow` is set; with that option, it stays pinned to the initially resolved root.

```bash
# Watch the latest active thread (auto-follows newly active threads)
t3-top

# Watch a specific thread by ID prefix or title substring
t3-top th-root-alpha
t3-top "feature implementation"

# Pin the initially resolved thread instead of switching when another thread updates
t3-top --no-follow

# Adjust polling interval (default: 0.5s)
t3-top --interval 1.0

# Limit recent counted activity history (default: 20)
t3-top --max-steps 15
```

### Snapshot mode

Print a single telemetry snapshot to standard output and exit:

```bash
# Print snapshot of the latest active thread
t3-top --once

# Print snapshot of a specific thread
t3-top th-root-alpha --once
```

### List threads

List recent application threads with model, status, subagent counts, and rollup token totals:

```bash
# Formatted Rich table
t3-top --list

# Machine-readable JSON array of thread summaries
t3-top --list --json
```

### JSON export

Export detailed telemetry for a selected thread and its full subagent tree:

```bash
t3-top --json
t3-top th-root-alpha --json
```

### Custom userdata directory

By default, `t3-top` reads from `~/.t3/userdata`. You can override this location:

```bash
t3-top --t3-dir /path/to/custom/userdata --once
```

### Terminal navigation and controls

Live watch mode provides an interactive terminal browser with full keyboard navigation across root application threads and subagent trees:

- Up / Down, k / j, or W / S: Move selection in the home Chats list or Usage periods, select top threads on a period screen, select activity entries on overviews or scroll an open activity detail, or change subagent selection when the subagent pane is focused.
- Tab / Shift+Tab: Move focus between Usage and Chats sections on the home screen, between Activity and Subagents selection on detail screens, or between the actions list and page on action details.
- E or Enter: Open the detail screen for the selected app thread, open the period screen for a selected usage period, open a top thread from a period screen, drill down into a highlighted child subagent, or open the focused activity detail. In a launch detail, E or Enter opens the linked child when available. E or Enter in an ordinary activity detail does nothing.
- Escape: Return one level, restoring focus, selection, and scroll position. From a period screen, Escape returns home with the period selection preserved. A child opened from a launch returns to that launch detail first.
- q or Ctrl-C: Exit t3-top cleanly and restore the terminal to canonical cooked mode.

#### Screens and breadcrumbs

The browser supports hierarchical navigation across primary view levels:

1. Home screen: Combines real-time workspace activity, aggregated token usage, and recent chats:
   - Live card: Displays the active root thread's status (`RUNNING`, `IDLE`, etc.), title, model, driver, and context window gauge bar. Also displays the turn clock ("Turn running <duration>" for an active turn or "Last turn HH:MM"), today's token total for that thread, latest recorded activity description (`Act: ...`), active subagent count with titles, and the count of other concurrently running threads. When no threads are active, reports "No active app threads".
   - Usage section: Displays token consumption across local calendar periods: Today, This week (starting Monday), and This month, along with each period's date range. Totals sum input + cache read + cache creation + output tokens across all threads. The usage report is cached and refreshes at most every 5 seconds, in the background so keys stay responsive; a failed read is retried after the same 5 seconds. Pressing Tab toggles focus between the Usage and Chats sections; Up / Down selects a period, and Enter opens the period screen.
   - Chats section: Displays root application threads ordered by most recent update, with model labels and status-colored titles.
2. Period screen: Detailed breakdown for a selected calendar period (Today, This week, or This month):
   - Usage over time: Stacked bar chart broken into hourly buckets (Today) or daily buckets (This week and This month; merged when terminal width requires). Bars stack provider tokens with distinct glyphs and colors: Claude (`█`, dark orange), Codex (`▓`, green), Antigravity (`▒`, medium purple), and Other (`░`, grey). A legend displays each provider's glyph, color, and total tokens. Future time buckets display `·`.
   - Providers table: Lists token breakdown by provider family with Input, Cache read, Cache write (cache creation), Output, and Total.
   - Top threads: Root threads ranked by in-period token usage, with descendant subagents rolled into the root app thread. Displays the root thread title, active provider families, and total tokens.
   - Navigation: Up / Down selects a thread; Enter opens the thread's detail screen with a breadcrumb starting with the period name (for example, `This week › Root Thread`). Pressing Escape returns to the period screen with the top thread selection preserved; pressing Escape again returns home with the period row still selected.
3. Detail screen: Displays the summary header, three-scope token metrics, context occupancy graph, immediate subagents table, and recent activity stream for the active thread.

When drilling down into child or native subagents, a breadcrumb header (for example, `Root Thread > Child Subagent > Grandchild` or `This week › Root Thread › Child Subagent`) indicates the active lineage depth. Pressing Escape ascends the stack one level at a time, preserving previous cursor selections and scroll positions.

#### Summary follow and selection pinning

The top telemetry header operates independently of the selected screen:

- In default live watch mode, the summary header automatically follows the latest active root thread across the workspace, while the detail body remains focused on whichever thread the user selected.
- Passing a thread query on the command line initially resolves and focuses that thread in the detail view.
- Passing `--no-follow` pins the summary header to the initial thread rather than following newly active threads.

#### Metric scopes and activity kinds

Token telemetry is reported across three explicit scopes:

1. Current turn: Metrics for the active or latest provider turn. If an active turn is still in-flight or the provider has not yet published token counts, current turn availability is reported as unavailable rather than inventing zero usage.
2. Whole chat · this agent: Metrics consumed directly by the inspected agent, excluding child activity.
3. Chat + subagents: Totals for the inspected agent and its descendants. Child activity stays separate from the agent's own activity.

Provider turns (Claude or Codex turn boundaries) and model inference steps (Antigravity steps) are tracked distinctly via `provider_turn_count` and `model_step_count`. Duration is the sum of recorded agent step and turn durations. Concurrent subagents can overlap, so this total is not wall-clock elapsed time.

For Claude/Codex, uncached input is now `inputTokens − cached − cache creation`, totals include cache creation, and `--json` steps and summaries include `new_input_tokens` and `context_read_tokens`.

#### Activity details and retention

Activity rows show a recorded description, total tokens, and duration. Visible ranges indicate the list window, and selection scrolls at the visible boundary. Open details to inspect the detail sections (Tokens with re-read vs. new split and end-of-turn context, What it did timeline, Prompt with `P` to expand, Details footer). Provider-turn totals cover the entire turn, including any listed tools. Missing usage differs from recorded zero; approximate or partial totals are marked.

Descriptions use recorded command, path, task, or explicitly associated message fields. Name-only tools and unassociated turns report unavailable details. T3 message associations require explicit references; arbitrary provider logs are not searched. Retained text is bounded and truncation is labeled. Recorded commands are displayed as text and never executed.

`--max-steps` retains recent counted activities, with known immediate-child launch entries merged separately. Launches do not add child tokens or duration to the parent's own metrics. Unknown launch times remain undated. Launch details remain inspectable when child telemetry is unavailable; navigation requires an available child. Child internal activity appears on the child's overview.

#### Context window occupancy and history

The context graph uses an automatic token axis, separate from recorded context capacity. The ceiling is the smallest 1, 2, or 5 times a power of ten covering displayed history and current usage. Zero and ceiling are labeled, with a midpoint when space permits. A full-height bar does not mean the context window is full. Capacity is labeled separately as Limit. Chronological buckets fit retained samples to the width; range shading preserves peaks above the latest value.

- Context provenance: Labeled as `Measured` when obtained from provider thread records, `Estimated` when reconstructed from Antigravity inference steps, or `Unknown` when omitted.
- Missing and default limits: Known context limits (such as 200k for Claude or 1M for Gemini) are displayed when recorded by the provider. If the limit is unspecified, it is reported as unknown rather than displaying arbitrary defaults as measured.
- Context drops: A decrease is labeled as a context drop. The graph retains the peak within its retained history; a drop alone does not establish that compaction occurred.
- In-memory retention: Context history is retained across live polling ticks and navigation switches using an in-memory LRU store bounded to at most 256 samples per agent for up to 64 recently viewed agents. Unchanged poll samples are deduplicated, and no persistent data is written back to upstream databases.

#### Color legend and status indicators

- Green: Running thread status or context occupancy below 60%.
- Yellow: Cancelled thread titles, context drops, or context occupancy between 60% and 85%.
- Red: Failed thread status or context occupancy exceeding 85%.
- Completed titles use bright blue; idle titles use bright white. Essential titles, values, axes, and controls remain bright. Selection has a contrasting background and a marker for terminals without color. Secondary metadata may be subdued.

#### Noninteractive behavior

In headless environments, CI pipelines, or when stdout is redirected to a pipe, t3-top detects the non-TTY stream and disables raw keyboard handling, printing a single snapshot or structured JSON and exiting immediately.

### Breakdown and estimated cost

The breakdown bar displays four token kinds: cache read, cache write, output (which includes thinking), and input. Press `$` or `c` on any screen to toggle every usage figure between token counts and estimated USD cost: home period totals, the lifetime total shown beside each chat, the period breakdown, chart, providers table and top threads (ranked by the active measure), each agent's usage lines and header total (agent plus all its subagents), the Subagents and Activity columns. The activity detail always shows both a step's tokens and its estimated cost. The context graph stays in tokens, since it measures window occupancy rather than spend. Press `g` to switch the period chart between provider and scope splits.

Dollars are estimates at API list prices from a static table in `src/t3_telemetry/prices.py` (checked 2026-11-05). On subscription plans this is what the same work would cost on the API. Unknown models are shown as unpriced tokens.

## Ghostty terminal integration

To run `t3-top` alongside T3 Code in a dedicated side pane, add a keybinding to `~/.config/ghostty/config`:

```text
keybind = ctrl+shift+t=new_split:right
```

Then launch `t3-top` inside the split pane:

```bash
t3-top --interval 0.5

```

The interface automatically scales across narrow split panes (60+ columns) and wide multi-monitor terminals.

## Data sources and the local store

`t3-top` aggregates telemetry from local agent session files and stores incremental token metrics in a local SQLite database.

### Native session files

The collector scans three native agent directories:

- **Claude.** Reads JSONL transcripts from `~/.claude/projects/*/*.jsonl` for root sessions and `~/.claude/projects/*/*/subagents/agent-*.jsonl` for subagents. Subagent descriptions are loaded from matching `.meta.json` files. The `CLAUDE_CONFIG_DIR` environment variable overrides `~/.claude`.
- **Codex.** Reads session rollout files from `~/.codex/sessions/**/rollout-*.jsonl` and `~/.codex/archived_sessions/**/rollout-*.jsonl`. The `CODEX_HOME` environment variable overrides `~/.codex`.
- **Antigravity.** Reads conversation databases from `<t3-dir>/providers/antigravity/*/antigravity-acp/conversations/*.db` and CLI conversations from `~/.gemini/antigravity-cli/conversations/*.db`. The collector inspects companion `-wal` files so sizes and modification timestamps reflect recent writes.

### Telemetry store and privacy

The store is a SQLite database located at `~/.t3-telemetry/telemetry.sqlite` by default. You can specify a different location with `--store`.

Only token counts and session metadata are stored. Tracked metrics include input tokens, cache read tokens, cache creation tokens, output tokens, thinking tokens, timestamps, model identifiers, tool names, session keys, and T3 thread bindings. Message text, prompts, completions, system instructions, and tool outputs are never written to the store.

The local store powers calendar usage views across Today, This week, and This month. Detailed thread overviews, turn metrics, real-time activity streams, and context occupancy graphs read directly from live T3 and Antigravity database files.

### Origin detection

Sessions in the store report their execution origin:

- **T3.** The session matches a known T3 Code app thread through a mirrored session binding.
- **cli.** The transcript indicates a command-line interface.
- **sdk.** The transcript indicates an SDK or client library.
- **unknown.** The session transcript contained no origin indicator.

Origin hints remain provisional until T3 supplies a binding.

### Store commands and options

Sync the telemetry store from local sources and exit:

```bash
t3-top --sync
```

Output includes `files_seen`, `files_read`, `steps_written`, and `errors`.

List native sessions recorded in the store:

```bash
# Formatted Rich table
t3-top --sessions

# Machine-readable JSON
t3-top --sessions --json
```

Specify a custom store path:

```bash
t3-top --store /path/to/custom/telemetry.sqlite
```

Disable the local store and read usage directly from T3 state:

```bash
t3-top --no-store
```

## Architecture

1. `protobuf.py`: Pure-Python wire parser decoding varints and packed length-delimited fields from Antigravity SQLite metadata blobs without external protobuf compilation dependencies.
2. `db_agy.py`: Reader extracting step timings, token counts, and tool names from Antigravity database files and transcript logs.
3. `db_t3.py`: Read-only SQLite client querying T3 `statev2.sqlite` projection tables for thread lineage, context window gauges, and turn usage records.
4. `store.py`: Persistent SQLite store recording incremental file offsets, native sessions, step token counts, tool names, and mirrored T3 bindings.
5. `sources/`: Native file parsers for Claude JSONL transcripts, Codex rollout logs, and Antigravity databases.
6. `collector.py`: Incremental discovery and ingestion engine syncing native agent files into the local telemetry store.
7. `t3_mirror.py`: Graph synchronizer mirroring T3 app threads and provider session bindings into the store.
8. `aggregator.py`: Tree aggregator resolving thread hierarchies, linking delegated subagents and native runs, and accumulating self and rollup metrics.
9. `ui.py`: Rich terminal renderables generating context gauges, summary metric comparisons, recursive subagent trees, and step stream tables.
10. `cli.py`: Command-line interface supporting live monitoring, snapshot printing, list queries, store sync, and JSON serialization.

## License

MIT
