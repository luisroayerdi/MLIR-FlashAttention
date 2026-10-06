# Contributing

This process applies to every contributor, human or AI agent.

## Before you start

Read `SPECS.md` (what the project is and must do) and `docs/LITERATURE.md`
(what already exists). If the work you have in mind isn't described in
`SPECS.md`, open an issue before writing code.

## Workflow

```
issue ──▶ maintainer triage ──▶ `ready` ──▶ branch + PR ──▶ review ──▶ maintainer merges
```

1. **Issue.** All work starts as a GitHub issue.
2. **Triage.** The maintainer labels it `ready`, or closes it with a reason.
3. **Branch and PR.** Branch `issue/<N>-<short-name>` from `main`. Open one
   PR per issue, using the description format below.
4. **Review.** The PR is reviewed against the checklist below. Fixes go on
   the same branch.
5. **Merge.** Only the maintainer merges, by squash: one commit per PR on
   `main`. Merging closes the issue.

Problems found inside an open PR are discussed in its review, not in new
issues. While the project has one maintainer, keep one PR open at a time.

## Issues

| Label | Use for |
|---|---|
| `proposal` | New work, or a change to `SPECS.md` (adding, changing, or dropping a feature) |
| `bug` | Something on `main` that is wrong |
| `finding` | Something on `main` that conflicts with the literature, the scope, or these standards |
| `ready` | Set by the maintainer: approved for work |

An issue states what, why, and which `SPECS.md` section it concerns. Bugs and
findings include evidence: a failing command, a file and line, or a source
link. Issues are closed, never deleted.

## Pull requests

Description format:

```markdown
Closes #<N>

## SPECS
<section(s) this implements, e.g. §3.2 Fusion>

## Changes
<what changed and why, a few lines>

## Prior work
<closest system in docs/LITERATURE.md, and why it isn't reused here>

## Tests run
<exact commands and results, or "Docs-only">
```

A PR:
- Does one thing: the issue it closes.
- Contains only what the issue requires. No speculative options,
  abstractions, or files.
- May edit `SPECS.md` only to move a feature it implemented from `planned`
  to `implemented`.
- Records non-obvious design decisions in `docs/DECISIONS.md`.

## Review

Reviewers rerun the tests themselves; they don't rely on the PR description.
Docs-only PRs (changes only `*.md` files) skip the build and tests.

1. **Spec.** Does the PR do what its issue and `SPECS.md` section describe,
   and nothing else?
2. **Scope.** Does it stay inside `SPECS.md` §2?
3. **Prior work.** Does something in `docs/LITERATURE.md`, upstream MLIR, or
   an existing tool already do this? Entries older than 60 days, or any claim
   of novelty, are rechecked against current sources.
4. **Tests.** Do the build, FileCheck tests, and numerical validation pass?
   Does new behavior have a test? Skipped for docs-only PRs.
5. **Standards.** Does the code follow the standards below?
6. **Size.** Could the same result be reached with less code?

Review comments start with a verdict line: `VERDICT: APPROVE` or
`VERDICT: CHANGES REQUESTED`, then the checklist, then line comments.

| Label | Meaning |
|---|---|
| `needs-review` | PR ready for review |
| `reviewed:approve` | Reviewer approves; waiting for the maintainer |
| `reviewed:changes` | Reviewer requests changes |
| `needs-maintainer` | Two review rounds without approval; maintainer decides |
| `maintainer:changes` | Maintainer requests changes |

## Coding standards

**C++ and TableGen** follow the
[LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) and
MLIR conventions:
- File header in the LLVM `//===- … -===//` style, stating the pass's purpose
  and any precondition it relies on.
- Every pass declares its dependent dialects and has a FileCheck test in
  `test/Attention/`.
- Prefer upstream MLIR utilities (rewriters, interfaces, existing passes)
  over hand-written equivalents.
- Preconditions are verified in code where possible; when not, they are
  stated in the file header and in `docs/DECISIONS.md`.

**Python** (`test/`, `benchmarks/`):
- Standard library and NumPy only, unless an issue approves a dependency;
  add it to `test/numerical/requirements.txt`.
- Each script has a module docstring with its purpose and usage.

**Both:**
- Lines under 80 characters.
- Comments explain why, not what; match the density of the surrounding code.
- Don't disable tests or warnings to make a change pass.
- Numbers reported in documentation come from scripts, never typed by hand.

## AI-assisted contributions

AI agents contribute through the same process. The maintainer is
accountable for every merged change and reviews each one before merging.
Agent roles are defined in `.claude/agents/`; agent-specific guidance is in
`AGENTS.md`.
