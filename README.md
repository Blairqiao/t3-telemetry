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

# Limit recent step history in the stream table (default: 20)
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

## Architecture

1. `protobuf.py`: Pure-Python wire parser decoding varints and packed length-delimited fields from Antigravity SQLite metadata blobs without external protobuf compilation dependencies.
2. `db_agy.py`: Reader extracting step timings, token counts, and tool names from Antigravity database files and transcript logs.
3. `db_t3.py`: Read-only SQLite client querying T3 `statev2.sqlite` projection tables for thread lineage, context window gauges, and turn usage records.
4. `aggregator.py`: Tree aggregator resolving thread hierarchies, linking
   delegated subagents and native runs, and accumulating self and rollup metrics.
5. `ui.py`: Rich terminal renderables generating context gauges, summary metric comparisons, recursive subagent trees, and step stream tables.
6. `cli.py`: Command-line interface supporting live monitoring, snapshot printing, list queries, and JSON serialization.

## License

MIT