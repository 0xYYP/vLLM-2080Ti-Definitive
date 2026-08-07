# AGENTS.md

This file governs the whole `vLLM 2080 Ti Definitive Edition` repository.

## Project Identity And Credit

This repository is a hardware-focused fork for dual RTX 2080 Ti / SM75 vLLM
serving. It builds on upstream vLLM and preserves the local runtime work,
profiles, documentation, and benchmark evidence needed to reproduce the 2080 Ti
stack.

If you publish, redistribute, repackage, benchmark, or build a derivative from
this repository, keep clear credit to:

- Upstream vLLM and its original license.
- `vLLM 2080 Ti Definitive Edition`.
- The repository author: `github.com/weicj`.

Do not remove existing attribution, license notices, benchmark provenance, or
project identity text. If you maintain a public derivative, state that it is
based on this project unless the code has been independently replaced.

## Upstream Compatibility

This project remains a fork of upstream vLLM. When changing source files that
come from upstream vLLM:

- Preserve upstream license and copyright notices.
- Prefer small, reviewable patches over broad rewrites.
- Keep SM75/Turing-specific behavior guarded or clearly documented.
- Do not present fork-specific behavior as upstream vLLM behavior.
- If an upstream `AGENTS.md` or contribution instruction applies in a copied
  upstream subtree, follow it as well.

## Maintenance Language

- Always use `$maintain-vllm-2080ti` when preparing or reviewing commits,
  branches, issues, pull requests, releases, changelogs, documentation,
  upstream synchronization, or other publication and maintenance work.
- Use Chinese as the primary maintenance language while preserving English
  technical conventions, upstream compatibility, and an accurate English
  entry point for international users.
- Do not rewrite existing or imported upstream history solely to translate it.

## Commit Message Discipline

- Use a commit body for every non-trivial local change, including behavioral,
  protocol, runtime, compatibility, or multi-file fixes, and whenever the root
  cause, safety boundary, or validation is not obvious from the subject alone.
- In the body, state why the change is needed, what behavior and boundaries
  result from it, and which validation was actually run. Keep model, hardware,
  KV precision, MTP, context, and benchmark details when making support or
  performance claims.
- A subject-only commit is acceptable only for a genuinely trivial mechanical
  change whose intent and effect are fully explained by the subject and diff.
- Before committing, inspect the complete subject and body with
  `git show -s --format=fuller <commit>`; do not treat a concise subject as a
  substitute for a useful maintenance record.
- Commit message formatting (repository-owner preference, enforced since
  2026-08-07): subject is a single line; the body is written as continuous
  paragraphs with NO manual hard line breaks — do not wrap sentences by hand
  (let the terminal/editor wrap naturally). Separate paragraphs with a single
  blank line. Verify with `git show -s --format=%B <commit>` that the body
  contains no hand-wrapped lines (only paragraph separators and list items).

## Runtime And Profile Rules

This repository is organized around validated runtime routes, not generic
benchmark guesses.

- Do not invent context-size, throughput, or support claims without evidence.
- Keep profile files focused on route parameters only. Do not store global
  service settings such as GPU selection, port, chat template, or reasoning
  defaults inside route profiles.
- Use `profiles/README.md`, `profiles/README.zh-CN.md`, and
  `docs/model-profile-routes.md` as the source of truth for shipped profiles.
- If adding or promoting a profile, include capacity evidence and throughput
  evidence using the repository's documented benchmark口径.
- Do not keep tiny smoke-only profiles as recommended deployment presets.

## Validation Before Publishing

Before committing or publishing changes, run the relevant subset of:

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile <changed-python-files>
git diff --check
```

For launcher/profile changes, also verify `launcher.sh --print-config` for the
affected route and mode. For runtime kernel or graph-policy changes, include a
real benchmark or smoke result that proves the changed path still works.

## Documentation Discipline

- Keep English and Simplified Chinese documentation consistent when both exist.
- Keep benchmark numbers tied to the exact model, KV precision, MTP setting,
  context, and benchmark method.
- Restore or update linked assets when moving documentation. Broken benchmark
  figures are treated as documentation regressions.
- Avoid overstating support. Use precise wording such as `validated`,
  `supported`, `experimental`, or `not promoted` according to the evidence.

## Repository Hygiene

- Do not commit local caches, model weights, logs, temporary workspace state,
  run outputs, or generated native build artifacts.
- Keep `README.md`, `README.zh-CN.md`, `CHANGELOG.md`, `VERSION`, and
  `pyproject.toml` version fallback aligned for releases.
- Release tags and GitHub Releases are separate. Pushing a tag is not enough to
  update the GitHub Release page.

## Merge And Release Workflow

The default branch `main` is protected: `required_status_checks`
(`shell-contracts`, `python-policy-tests`), `enforce_admins`, and force pushes
are disabled. All changes land through pull requests.

Standard flow for each change:

1. Branch from `main`: `git checkout -b fix/<topic>`.
2. Commit with conventional subjects and a full body per the commit message
   discipline above.
3. Push the branch: `git push -u origin fix/<topic>`.
4. Open a PR against `main` and fill the description following the project
   template (background/root cause, goals, implementation, validation,
   test environment, risks).
5. Wait for CI to go green.
6. Merge with **Rebase and merge** so each commit stays independent and `main`
   history remains linear. Do not use squash (it collapses the history) and
   avoid merge commits (they create a fork topology).
7. Delete the remote branch after merge.

Rules that prevent history rewrite problems:

- Do not rewrite pushed branch history unless the repository owner explicitly
  approves it. The original rationale (force push breaks CI diff-base
  resolution) was fixed by `3b7d503` (cpu-validation falls back to `HEAD^`
  when `github.event.before` is unreachable), so rewriting is technically
  safe; the remaining cost is confusing merge records for other clones.
  Before force pushing: confirm no other clone has dependent work, announce
  the rewrite, and have other clones `git fetch` + `git reset --hard
  origin/<branch>` to align.
- Prefer rewriting only commits that are not yet pushed; for pushed commits,
  treat rewriting as the exception and get explicit owner approval (as done
  for the 2026-08-07 commit-message format rewrite).
- If a commit message needs fixing after push and rewriting is not approved,
  add a follow-up commit or open a new PR instead.
- Do not bypass `main` branch protection to rewrite merged history. If the
  merged topology is wrong, prefer a corrective PR over a force push.
