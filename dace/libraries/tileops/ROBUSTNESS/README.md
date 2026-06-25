# ROBUSTNESS notes — how this works (read after PROGRESS.md)

This directory lets any agent resume the "robust cuTile expansions" effort with full
context, even across usage-limit interruptions.

## Files

- `PROGRESS.md` — master tracker: goal, env facts, status table, decision log, activity log.
- `README.md` — this file: the coordination protocol + note-keeping rules.
- `SHARED_FINDINGS.md` — index of cross-cutting knowledge.
- `CURRENT_STATE.md` — survey of what each node's cuTile expansion does *today* + gaps (auto-filled by a survey agent).
- `CUTILE_API_NOTES.md` — distilled facts from the NVIDIA cuTile-Python docs (auto-filled by a survey agent).
- `node_<name>.md` — per-node running log. Created when work on that node starts.

## Roles

- **Human** — the real user. Owns big/strategic decisions. The coordinator relays to them.
- **Coordinator** — top-level Claude. Spawns orchestrators, plays "user" to each, keeps
  notes, relays decisions. Keeps its own context lean (delegates heavy reading).
- **Orchestrator (one per node)** — a `coding-orchestrator` subagent driving research →
  plan → implement → test → review for that node. Its AskUserQuestion may not reach the
  human; it must instead surface questions in its messages so the coordinator can relay.

## Protocol

1. Coordinator picks the next node per the agreed ordering, creates `node_<name>.md`,
   sets status in `PROGRESS.md`.
2. Coordinator spawns the orchestrator with: the goal, the node's file path, env facts,
   pointers to these notes, and an explicit instruction to (a) NOT rely on AskUserQuestion
   reaching a human — instead list decisions/questions plainly in its reply, and (b) write
   end-to-end GPU runtime tests, not just structure checks.
3. Coordinator relays orchestrator questions to the human via its own AskUserQuestion,
   then feeds answers back via SendMessage to the same orchestrator (preserves context).
4. On meaningful progress (plan agreed, tests passing, blocker), coordinator updates
   `node_<name>.md` and the status table. **Update notes BEFORE long/expensive steps**, so
   an interruption never loses the thread.
5. When a node is DONE (runtime tests pass), coordinator records the final test command +
   result in `node_<name>.md` and marks DONE.

## Note-keeping rules

- One concern per file; keep PROGRESS.md's table authoritative.
- Always write absolute, reproducible test commands (with CWD).
- Convert relative dates to absolute (today is 2026-06-24).
- If you discover a *core* DaCe bug, also follow the repo rule: entry in `dace/CORE_BUGFIXES.md` + reproducer.
