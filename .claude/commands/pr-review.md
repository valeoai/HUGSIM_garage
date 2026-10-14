---
description: Multi-agent PR review in rounds; pr-reviewer agents review through separate lenses, findings are verified, and with --fix they are fixed between rounds
argument-hint: "[PR number | PR URL | branch] <low|medium|high|max> [--model opus|sonnet|haiku|fable] [--fix] [--rounds N] [--gpu vai:<job id> | jz:<GPU-hours>]"
disable-model-invocation: true
---

Run a multi-round review of a pull request. You are the orchestrator: you read the PR, dispatch
`pr-reviewer` agents (`.claude/agents/pr-reviewer.md`), merge and verify what they find, have the
findings fixed if asked, and report. You do not review the code yourself, beyond checking what
the agents report.

Arguments: `$ARGUMENTS`

**`gh` resolves to upstream here.** This checkout has two remotes, `origin` (valeoai/HUGSIM_garage)
and `upstream` (hyzhou404/HUGSIM), and `gh`'s default repo is the upstream one. Pass
`-R valeoai/HUGSIM_garage` to every `gh` command, unless the target is a hyzhou404/HUGSIM URL.

## 1. Parse the arguments

- **target** (optional): a PR number (of valeoai/HUGSIM_garage), a PR URL (of this fork, or of
  hyzhou404/HUGSIM for a fix proposed upstream), or a branch name. If there is none, review the current
  branch against `main`.
- **effort** (required): `low`, `medium`, `high` or `max`, given as a bare word or as `--effort X`. If it
  is missing, ask with AskUserQuestion. Do not guess it.
- **`--model`**: `opus` (the default), `sonnet`, `haiku` or `fable`. It is passed as the Agent `model`
  of every agent you dispatch. Your own model does not change.
- **`--fix`**: confirmed findings are fixed between rounds. Without it, nothing in the tree is edited.
- **`--rounds N`**: the maximum number of review (and fix) rounds, 1 by default. A closing review, which
  fixes nothing, always runs after them (section 5).
- **`--gpu`**: short closed-loop checks of the tree (section 4f). `vai:<job id>` runs them in a running
  vai job of the user's; `jz:<GPU-hours>` submits them on Jean Zay within that budget. Without it,
  nothing runs on a cluster.

Reject anything else and print the usage line from `argument-hint`.

## 2. Resolve the target and set up the run

1. If the target is a PR, run `gh pr view <target> -R valeoai/HUGSIM_garage --json number,title,body,url,baseRefName,headRefName,headRefOid,isCrossRepository,commits,files,closingIssuesReferences`
   (no `-R` for a URL), and `gh issue view <n> -R <its repo> --json title,body` for each linked issue (and
   any issue the body names as `#<n>`), since they define what the PR has to do. The base remote is
   `origin` for this fork's PRs and `upstream` for hyzhou404/HUGSIM's.
   For a branch, or for no target, use `origin/main` as the base and the branch as the head. Run
   `git fetch <base remote> <baseRefName>` first, and `git fetch upstream main` too: the contracts lens
   checks upstream-bound fixes against it. Stop if there is nothing to review: no commits past the merge
   base and no uncommitted changes.
2. Find the **working tree**: a checkout at the head, because the reviewers read the working tree.
   - If the head branch is checked out in a worktree (`git worktree list --porcelain`), use that tree;
     `gh pr checkout` would fail there anyway.
   - Else, if this checkout's `HEAD` is already at the head, use it as it is.
   - Else, if `git status --porcelain` is empty, run `gh pr checkout <n> -R valeoai/HUGSIM_garage` (or
     `git checkout <branch>`) here, and note the branch you left so the report can name it.
   - Else the tree is dirty: stop and ask whether to stash, commit, or review from a new worktree
     (`git worktree add ../HUGSIM_garage-review-<target> <branch>`).

   In whichever tree you use, uncommitted changes count as part of what is reviewed; say so if there
   are any. From here on, run every `git` and file command in that tree (`git -C <tree>`, absolute
   paths). Agents start in this session's directory, so the prompts name the tree.
3. `BASE=$(git -C <tree> merge-base <base remote>/<baseRefName> HEAD)`. Create the run directory
   `<session scratchpad>/pr-review/<target>-<YYYYmmdd-HHMMSS>/` and write `context.md` in it. Its first
   line is `Working tree: <absolute path>`. Then: the title, URL, body, the linked issues' titles and
   bodies, base branch, `BASE`, head SHA, the commit list (`git log --format='%h %s' BASE..HEAD`), the
   **PR files** (`git diff --name-only BASE HEAD`, fixed for the whole run), `git diff --stat BASE`, the
   effort, the model, whether `--fix` is on, the rounds, and the `--gpu` target. Add `Drive_gym checkout:
   <path>` if one exists next to this repo (`../Drive_gym`, or `../HUGSIM_gym`): it vendors this repo
   as `third_party/HUGSIM`, and the integration lens greps it for what it imports from here.
4. Split the changed files into **reviewed** and **excluded**. Excluded: `pixi.lock`, `*.egg-info/`, and
   binary or generated files (`*.ply`, images, videos, `assets/`); third-party code (`data/InverseForm/`
   but for its `infer_*.sh`, `submodules/simple-knn/`) is excluded too, and any change there is reported
   as a finding, under contracts. Count the changed lines in the reviewed files.
5. With `--gpu jz:<hours>`, set the budget now (the jz MCP's `budget_set`, for those hours). Never submit
   beyond it. Jobs bill the default account in `.claude/cluster/accounts.json`; this repo may not have
   one, and the jz MCP's rule then applies: ask the user, never create it unasked.

## 3. Effort sets the plan

| effort | lenses | verification | findings kept |
| --- | --- | --- | --- |
| low | correctness, contracts | you re-read each cited line | high confidence, severity medium or worse |
| medium | correctness, contracts, integration | you check each finding against the code | no nits |
| high | correctness, contracts, integration, design, docs | `pr-reviewer` agents in `verify` mode | CONFIRMED and PLAUSIBLE |
| max | all six: also runtime | `verify` mode, two verifiers for each critical/high finding | CONFIRMED and PLAUSIBLE, low-confidence ones included |

At `high` and `max`, if the reviewed files have over ~1500 changed lines, give the correctness lens two
reviewers, each owning half of the files (split by directory, so related files stay together: `sim/`
apart from `closed_loop.py` and the scripts, and the configs with whatever reads them).

Lenses overlap on documentation, so one owns it: text in `README.md`, `data/README.md`, docstrings,
comments and argparse help belongs to the docs lens when it runs (to contracts otherwise). The other
lenses report a doc problem only as part of a code finding.

## 4. Rounds

For round `r = 1 .. N`:

### a. Prepare

1. Regenerate the reviewed files: `git diff --name-only BASE` plus untracked files
   (`git ls-files --others --exclude-standard`), minus the excluded ones. Fixes may have touched files
   the PR did not.
2. In `round-<r>/`, write `diff.patch` (`git diff BASE -- <reviewed files>`, which includes uncommitted
   fixes from earlier rounds) and `prior.md`: every finding from earlier rounds with its id, file:line,
   summary and status (`fixed`, `rejected` and why, `open`, `skipped` and why, `deferred`), and for
   `--fix` the fixes made so far (`git diff HEAD`, saved as `round-<r>/fixes.patch` and referenced from
   `prior.md`). Point to `facts.md` and `gpu_tests.md` in the run directory if they exist. In round 1,
   `prior.md` says "none".

### b. Review

Dispatch one `pr-reviewer` agent per lens (or per lens and file half), **all in a single message**, with
`model: <model>`. The prompt for each one:

```text
Mode: review. Lens: <lens>. Effort: <effort>. Round <r> of at most <N>.
Working tree: <abs path>. Read files and run git there (git -C <tree>, absolute paths).
Run directory: <abs path>. Read context.md, round-<r>/prior.md and round-<r>/diff.patch there,
and facts.md and gpu_tests.md if they exist.
Files you own: <all reviewed files | the list>.
<if the docs lens runs and this is not it: "Doc text is the docs lens's; report a doc problem only as
part of a code finding.">
<round > 1: "Earlier rounds' findings and fixes are in prior.md. Check that each fix is right and
broke nothing, and look for what earlier rounds missed: start with the files that got the fewest
findings.">
Report in the JSON format your instructions define.
```

Wait for every reviewer to finish. If one fails or returns no parsable JSON, send it the error with
SendMessage once (load it with ToolSearch first); if it fails again, go on without it and say so in
the report.

### c. Merge

Collect the findings. Merge duplicates (same root cause: same file and nearby lines, or one bug seen
through two lenses) and keep the clearest wording and the highest severity. Drop findings on excluded
files (except "the PR edits third-party code"), findings the effort filter excludes, and anything
`prior.md` already lists as rejected unless the reviewer gives new evidence. Give each kept finding a
stable id `R<r>-<n>`.

### d. Verify

1. **Cluster facts.** Some findings turn on a fact only a cluster holds (a scenario or scene file, an
   AD launcher, a job's log, an episode folder); reviewers mark them `needs_fact`. If the jz-cluster or
   vai-cluster MCP tools are available, read those facts yourself, read-only: on Jean Zay `read_file`,
   `list_dir`, `find_files`, `get_job`, `job_logs`, `job_history`, `list_jobs`, and `run` only for
   commands that read (`cat`, `ls`, `grep`); on vai `get_job`, `job_logs`, and `job_exec` only for
   commands that read, in a job already running. Never submit, cancel, write or remove anything for a
   fact. Append each fact with its source (path, job id) to `facts.md` in the run directory and give it
   to the verifiers. A fact you could not read leaves its finding PLAUSIBLE, naming the fact.
2. **Hard evidence.** At any effort, you may settle a finding yourself when hard evidence decides it: a
   `git merge-tree` result (an upstream-bound fix against `upstream/main`, say), a job log, a fact from
   `facts.md`, a command's output. Record the evidence as the reason; the finding needs no verifier.
3. **The rest**, as the effort table says. At `high`/`max`, group the findings by file into at most four
   batches and dispatch one `pr-reviewer` per batch in `mode: verify` (two per critical/high finding at
   `max`, in different batches), all in one message, with the findings' JSON in the prompt, the working
   tree and the run directory. A finding is CONFIRMED if its verifiers confirm it; at `max`, if two
   verifiers disagree, read the code yourself and decide. At `low`/`medium`, read the cited code yourself
   and drop what does not hold up.

REJECTED findings go into the next `prior.md` with the reason.

### e. Fix (only with `--fix`)

First decide whether this is the **last fix step**: round `N`, or a round that will stop early (4g's
rule, which depends only on this round's verdicts).

Fix only CONFIRMED findings. PLAUSIBLE ones are reported, not fixed. Of those:

- **Nits wait for the last fix step**, and are fixed there all at once (with the nits deferred by earlier
  rounds that are still valid). Fixing nits round after round mostly creates new nits. Until then their
  status is `deferred`.
- **Scope.** A fix that touches a file outside the PR files (`context.md`), or changes what the env, the
  rendering, the actors, the tracker or the scorer do (only a cluster run can check that, and it may move
  scores), is made only for findings of severity medium or worse. Below that the finding gets outcome
  `skipped`, "outside what an automatic fix should change", and is reported. Docstring and comment fixes
  in those files are fine.

Then:

1. Group them so that no two groups touch the same file: a finding's files are its `files_to_fix`, and
   findings that share a file go in the same group. Merge groups until there are at most four. A finding
   whose edits split cleanly by file (code in one group, README in another) may be split across groups.
2. Dispatch one `pr-reviewer` per group in `mode: fix`, all in one message, with `model: <model>`, the
   group's findings in JSON, the files it owns, the working tree and the run directory. Fixers run in
   parallel, which is safe only because their files are disjoint. Keep it that way.
3. When they are done, read `git diff` yourself. A fix that is wrong or reaches beyond its finding: correct
   it or undo it with Edit, and record that. Never `git checkout`, `restore`, `reset` or `stash`; they
   would also throw away the user's work. Then check every file touched this round in the working tree:
   `python -m py_compile` on the Python files, `bash -n` on the shell scripts, a YAML parse on the configs
   (`python -c 'import yaml, sys; yaml.safe_load(open(sys.argv[1]))' <file>`). There is no pre-commit
   here; make those pass.
4. Do not commit or push.

### f. GPU checks (only with `--gpu`)

Run them after a fix step that touched Python, shell or config files under `sim/`, `scene/`,
`gaussian_renderer/`, `utils/`, `submodules/Pplan/`, `configs/sim/`, or `closed_loop.py`, `get_score.py`
or `run_*.sh`; without `--fix`, run them once, before round 1. They turn "plausible" into "ran"; they are
not a benchmark sweep.

1. **What to run:** every `gpu_check` the open findings carry, plus one `closed_loop.py` episode for each
   entry point the PR or the fixes changed, with the PR's new flags, on the smallest scenario (a nuScenes
   `easy` one). Run it at the published operating point (`configs/sim/kinematic.yaml`, dt 0.25) and, if
   the PR touches anything timed (dt, the episode cap, the actors, the tracker), at
   `configs/sim/kinematic_10hz.yaml` as well. Keep the native scorer on unless the PR is about
   `--skip-native-eval`, so `eval.json` is written. If the PR claims behaviour is unchanged at dt 0.25,
   run the same episode from a second copy at `BASE` and compare the two `data.pkl` (`ego_box` and
   `obj_boxes` per frame) and `eval.json`: a difference is a finding.
   - **The AD side.** The real agents live in other checkouts with their own environments. Unless the PR
     concerns one whose launcher (`HUGSIM_AD_<NAME>`) the user has on that cluster, use a stand-in that
     drives straight ahead, written into the copy's `outputs/pr_review_ad/` and passed as `--ad stub
     --ad_path <copy>/outputs/pr_review_ad/stub.sh`. `stub.sh` is `exec python "$(dirname "$0")/stub.py"
     "$2"`, and `stub.py`:

     ```python
     import os, pickle, sys, time
     import numpy as np
     out = sys.argv[1]
     obs_pipe, plan_pipe = os.path.join(out, 'obs_pipe'), os.path.join(out, 'plan_pipe')
     while not (os.path.exists(obs_pipe) and os.path.exists(plan_pipe)):
         time.sleep(1)
     plan = np.stack([np.zeros(6), 2.0 * np.arange(1, 7)], axis=1)  # (right, forward), 4 m/s
     while True:
         with open(obs_pipe, 'rb') as p:
             msg = pickle.loads(p.read())
         if isinstance(msg, str) and msg == 'Done':
             break
         with open(plan_pipe, 'wb') as p:
             p.write(pickle.dumps(plan))
     ```

     Its plan is deterministic, which is what makes a before/after comparison mean something.
2. **Where:** never in the user's own run tree, and never next to the scene data. Both clusters run
   HUGSIM from Drive_gym's tree, whose pixi env is built for it. After sourcing its env script, replace
   `PYTHONPATH` with the review copy's (`<copy>/sim:<copy>`), so that `hugsim_env` and `sim.*` come from
   the copy and Drive_gym's vendored `third_party/HUGSIM` is off the path. Before running anything, check
   `python -c 'import hugsim_env, sim.utils.plan as p; print(hugsim_env.__file__, p.__file__)'` from the
   copy's root: both must be inside the copy.
   - `vai:<job id>`: rsync the working tree to `~/workspace/HUGSIM_garage-review-<target>` on the
     cluster NFS (`--exclude .git --exclude .pixi --exclude output --exclude outputs`). In the job:
     `source ~/workspace/HUGSIM_gym/env.sh`, then `export PYTHONPATH=<copy>/sim:<copy>
     HUGSIM_DATA=$HUGSIM_DATA_ROOT HUGSIM_OUT=<copy>/outputs/`, and `cd <copy>`. Run the checks with
     vai-cluster `job_exec` in that job (check `nvidia-smi` first; the job may see its GPU as another
     index), `--ad_cuda` on the same GPU.
   - `jz:<hours>`: copy the working tree to `$DRIVE_GYM_SLOT/garage_review-<target>/` (the slot is
     `/lustre/fsn1/projects/rech/glq/$USER/hugsim_gym`). Put the checks in one `#!/bin/bash -l` sbatch
     script on `qos_gpu_h100-dev`, with absolute paths, that sources the slot's `tree/jz/env.sh`
     (Drive_gym's: modules, CUDA 12.4.1, caches, HUGSIM's pixi env) and then exports the same
     `PYTHONPATH`, `HUGSIM_DATA` and `HUGSIM_OUT` as on vai; submit it with the jz MCP within the budget
     set in step 2. Jean Zay's scenario tree has an older layout than vai's: find the scenario with
     `find_files`, and check that the base config's `model_base` and `realcar_path` resolve under
     `$HUGSIM_DATA` before submitting.
3. **Read the logs**, the episode's `output.txt` (the AD side's) included: a clean exit is not a pass,
   and `closed_loop.py` prints an exception's traceback and exits 0. Append a section to `gpu_tests.md`
   in the run directory: host and job, which rounds' fixes the tree holds, and for each check the
   command, the exit code, the number of steps driven and why the episode ended, `eval.json`'s scores,
   the before/after comparison if one ran, and the finding ids it settles. Write it before the next
   round's `prior.md` or the closing review's dispatch, so those reviewers see it.

A check that fails is a finding (severity high unless the log shows otherwise), handled in the next
round, or in the closing review after the last round.

### g. Close the round

Write `round-<r>/summary.md`: the findings with their verdicts and outcomes, the reviewers that failed,
the files touched, the GPU checks. Then stop early if:

- with `--fix`: no finding this round was CONFIRMED with severity medium or worse. Its lows and the
  deferred nits were fixed in 4e, and the closing review reviews those fixes.
- without `--fix`: round `r > 1` found nothing that earlier rounds had not already found.

## 5. Closing review

This step always runs, even after an early stop. It reports what work is still missing. It fixes
nothing, whatever `--fix` says.

1. Write `closing/diff.patch` and `closing/prior.md` as in 4a, with the outcome of every round, and make
   sure `gpu_tests.md` and `facts.md` are up to date.
2. In one message, dispatch:
   - one `pr-reviewer` with `Lens: completeness`, `Mode: review`, owning all reviewed files;
   - if the last round made fixes, the effort's lenses again, told to review the fixes in
     `closing/prior.md` (and the code around them) first: those fixes have not been reviewed yet.
     If the last round made no fixes, the code it reviewed is unchanged, so skip them.
3. Merge and verify the findings as in 4c and 4d (ids `C-<n>`). `needs-run` findings need no
   verification. Do not fix anything: whatever is confirmed here is the remaining work.

## 6. Report

1. If the ReportFindings tool is available, call it once, most severe first, with: every finding still
   open, skipped, deferred or PLAUSIBLE (from the rounds and the closing review), and the fixed findings of
   severity low or worse. Fixed nits are only counted, in the summary. The tool takes at most 32; if
   there are more, keep the most severe and say in the summary where the rest are (the rounds'
   `summary.md`). Pass `level: <effort>`, `verdict` (CONFIRMED/PLAUSIBLE), `outcome` with `--fix`
   (`fixed`, `skipped`, `no_change_needed`), `category` = the lens, and a `failure_scenario`. Do not
   also print the findings as text. Without the tool, print them as a list ranked by severity, each with
   a [file:line](path#Lline) link, the failure scenario and its status.
2. Then **Remaining work**, from the closing review: a verdict on the first line, either "complete as
   far as this review can tell" or "work missing", then the missing items, the open or PLAUSIBLE
   findings, and the cluster runs the PR still needs, each one actionable.
3. Then a short summary: the target, head SHA and working tree, effort/model/rounds run (and why it
   stopped), the reviewers per round, the number of findings kept, rejected and fixed per round (fixed
   nits included), the GPU checks run, and the run directory. With `--fix`: `git diff --stat`, the
   result of the compile/parse checks, a reminder that nothing is committed, and what only a cluster run
   can check (a fix in the env, rendering, actors, tracker or scorer that no GPU check covered is not
   tested until an episode runs, and may move scores). If the PR changes what Drive_gym imports from
   here, say that Drive_gym takes it on at its next re-vendor. If you checked out a branch, name the
   branch the user was on.
4. Do not post anything to GitHub. Offer, in one line, to commit the fixes or to post the findings as PR
   comments (on valeoai/HUGSIM_garage, with `-R`; never on upstream unless the user names it).
