## mission
the central aim of this project is optimizing inference for flux 2 klein (4b/9b) on jax/tpu for scale. it serves as both a learning, utility and portfolio project. the goal is a jax/flax port that scales up on google tpus.

## agent-hub
this project is registered in the shared agent hub at `/home/tensor/Shared/code/agent-hub`. it is a file-based coordination space shared across projects — read it before starting substantial work, and record handoffs before stopping.

- **read first:** `INDEX.md` (task roster + status), then this project's task record in `tasks/`, then any messages addressed to it in `messages/`.
- **the task record for this project** is `tasks/20260920-flicker-flux2-klein-jax-port.md`. keep its `Status:` to one of the five enum values (`queued` | `active` | `blocked` | `done` | `superseded`) — prose statuses break the index generator.
- **handoffs** go in `messages/`, named `<task-id>-<slug>.md` or a UTC timestamp plus a unique suffix.
- **reusable findings** with evidence and dates go in `wiki/`; **outputs** (reports, checkpoints, generated audio worth pointing at) are registered in `artifacts/`.
- always write the canonical path `/home/tensor/Shared/code/...`; `/home/tensor/code` is a symlink to it, and an exact-string search only matches the spelling given.
- the hub is its own git repo outside the project roots. commit record changes — the hub is shared memory, not an event bus, and **writing a message does not wake another session**.

## code explanation
- always create a project.md file(check docs folder) where you explain the components of the code thoroughly, including the interlinking relationships. Don't rewrite the whole thing everytime, just look through and edit/add as needed, and keep it minimal. you can also use it as your reference in order to look through and have a birds eye view of the code structure, so keep it as your jotter, sort of. it should be structured in such a way that I could use it to understand the whole project if I am lost.
- create a log of edits per message, or rather, edit session, where you state in concise format what was changed/added/edited, in logs.txt, whichever you prefer
- add comments explaining each function and class and how they work. the comments should be explanatory enough (but still concise) so I can reimplement/refactor the code
- when making a fix, explain thoroughly why and what caused it. then TELL ME how you would fix it, unless I tell you to leave the fixing to me. tell me what broke exactly before you fix the code yourself.
- **this laptop has no gpu.** model download and inference runs go to **Modal (A10G)** — victoria is a 6 GB card and klein needs ~15 GiB, so it cannot hold this model at all. do not run gpu or memory-intensive scripts locally unless explicitly told to.

## working style
- never refactor something the user doesn't understand. ensure I understand it before changing, explain well and further explain if asked.
- give teaching-style explanations, then let the user predict the code before reading it.

## notes
- keep the code very, very minimal, as lean as possible. do not bloat it.
- check for code references in `refs/` if/where necessary. `refs/flux2` is a **read-only upstream checkout** — do not modify it; patch behaviour in our own scripts instead.
- write core logic as pure functions with immutable data models
- ensure state changes flow through function returns, not side effects
- avoid excessive comments, the code should be self-documenting
- use lowercase for all log messages and print statements
- design minimal utility functions that can be easily composed, avoid constant usage of try/except, and numerous if/else checks
- use consistent naming patterns for similar operations
- use `uv run` to execute python scripts, where applicable.

For type annotations:

- add complete type hints for all parameters and return values
- use pipe syntax for union types (`str | None` rather than `Optional[str]`)
- apply `Literal` for constrained string values when appropriate
- define type aliases for complex types at the module level
- always specify return types, including `None` when a function returns nothing

When organizing imports:

- follow order: standard library → third-party → local modules
- use these standard abbreviations:
  - `import pathlib as pl`
  - `import datetime as dt`
- import from specific modules rather than packages

- when writing articles/reports, keep it minimal, make sure every word there justifies its own existence.
- write notes in one persistent markdown file as you go, where you log your findings, so you dont lose complete context