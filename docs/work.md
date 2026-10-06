# Work: one session, one worktree, one branch, pushed always

`tatbot work` is how changes reach `main` on a development machine. Every
session works in a worktree of its own, on a branch that is pushed as soon as
it exists and after every commit. Landing is one command that rebases, runs
the land gate and pushes to `main`. A session cannot end with work that is
neither landed nor parked with a reason.

## Why

Two or three agents used to share one checkout and were told to "pull
--rebase and push often". A shared tree is never clean, so the rebase always
refused. Agents improvised with cherry-picks into throwaway worktrees, which
put every landed commit on origin under a new SHA while the original stayed
on local `main`. After two weeks that local `main` was 25 commits ahead and 89
behind, eleven "dirty" files were stale copies of landed commits nobody dared
touch, and a real fix sat unpushed for a day because a commit-time gate had
refused it. Work was durable only when someone remembered to land it.

Here work is durable the moment it exists.

## The commands

| Command | What it does |
| --- | --- |
| `tatbot work start --task NAME` | A worktree under `~/tatbot-work/` on branch `agent/<node>/<session>`, cut from `origin/main` and pushed at once. `--adopt` moves uncommitted changes out of the mirror checkout into it, cutting the worktree at the mirror's own revision so they apply as written; `land` rebases them onto `main`. |
| `tatbot work status` | Every worktree on this node: dirty files, commits ahead of `main`, unpushed commits, parked reason, idle time. Stale agent branches on origin with no worktree here. |
| `tatbot work land` | Rebase onto `origin/main`, push the branch, run the land gate (lint, disclosure, the export gate, complexity of the changed files: what the author answers for, under a minute), push `HEAD:main`, retrying the rebase while `main` moves, then delete the agent branch. `--full-check` runs the whole fast tier first; `--no-check` skips the gate; CI runs everything on `main` regardless. `--done` removes the worktree. After the push, the repository's post-land hook (`internal/ci/post-land`, when present) receives the landed range as `base head`; a private tree uses it for infrastructure that must react to what landed — here, rebuilding the isolated CI worker's images when the diff changed an input they are built from, so the worker does not sit at "rebuild and redeploy required" until someone notices. A failing hook fails the land's exit code; the push is never undone. |
| `tatbot work park "why"` | Leave the work unlanded on purpose: a `wip:` commit if the tree is dirty, pushed, with the reason recorded. |
| `tatbot work sweep` | What the hourly timer runs: autosave dirty worktrees idle for 30 min as `wip:` commits, push anything unpushed, move edits left in the mirror for a day into a pushed `mirror-rescue` worktree, fast-forward the mirror, prune worktrees that are clean and landed and idle for a day, delete remote agent branches already contained in `main`. |
| `tatbot work init` | Once per machine: mark this checkout as the mirror, arm the git hooks, install the user systemd timer. |

The shared checkout on each machine (the one `config/nodes.json` names as its `checkout`) is the **mirror**. Its
pre-commit hook refuses commits; the sweeper fast-forwards it; nothing is
edited there. Deployment reads from it, and so does `tatbot --on <node>`.

## What the hooks do

- **Claude Code `SessionStart`** (`.claude/settings.json`, tracked): creates
  the session's worktree and tells the agent where it is. A resumed session in
  its worktree is told its branch state instead.
- **Claude Code `Stop`**: refuses to end the session while the worktree holds
  uncommitted files or commits not on `origin/main`, unless it is parked. The
  message names the two ways out. After three refusals with no change it lets
  the session stop and the sweeper autosaves what is left.
- **git `post-commit`**: pushes the agent branch. Never fails the commit; an
  unreachable origin is reported and the sweeper retries.
- **git `pre-commit`**: lint and ratchets as before, plus a guard that refuses
  commits in the mirror.
- **git `pre-push`**: the public export gate, only for a push to `main`.
  Agent branches push free.

## Reading `tatbot work status`

```
tatbot work — dev1 2026-09-16T18:20:11Z
  mirror ~/tatbot: clean
  fix-leader-mount-20260916T181233Z         UNLANDED
    agent/dev1/fix-leader-mount-20260916T181233Z  dirty=2 ahead_of_main=3 unpushed=0 idle=14.0m task=fix leader mount
  remote agent/dev2/calib-20260915T090000Z (a1b2c3d) unmerged, 31.5 h old, no worktree here
  unlanded=1 parked=0 stale_remote=1
```

`unpushed=0` with `ahead_of_main=3` is the normal healthy state: the commits
are on the server, they just have not been landed. `tatbot status` carries the
same counts as the `unlanded_work` observation, so the fleet view shows which
node is sitting on work.

## Recovering work

Nothing in this flow deletes a commit. An autosaved or parked branch is on
origin under `agent/<node>/<session>`; check it out anywhere with
`git fetch origin && git worktree add ~/tatbot-work/<session> agent/<node>/<session>`
or resume it on its node with `tatbot work start --session <session>`. A
branch already contained in `main` is deleted by the sweeper an hour after it
merged; a branch with unlanded commits is never deleted by anything here.

## Conventions

- Commit in your worktree the way you always did: small, Conventional Commits.
  `wip:` commits are what the sweeper and `park` make; squash or reword them
  before landing if they would be noise on `main`.
- Land before you stop. Park only with a reason another person could act on.
- Run `scripts/check` (the fast tier), `scripts/check rust` or `scripts/check
  sim-fast` in the worktree as the change warrants; `land` runs only the gate
  the author answers for, so that someone else's red job cannot strand a change.
- A worktree costs a checkout of the tree, not a build: uv environments,
  `node_modules` and `rust/target` are per worktree and start cold. For a
  Rust or simulator task, keep the worktree for the whole task rather than
  starting a new session per step.
