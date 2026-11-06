---
name: ggml-fork
description: Add a fix or a speed-up to the ggml fork speech.cpp builds (the branch speech.cpp of github.com/nyosegawa/ggml) and move speech.cpp's submodule to it. Covers the fork's branch and commit order, test-backend-ops cases first, showing a fix's cases fail without it on Metal and on Vulkan through CI, measuring a speed-up, the pull request that moves the submodule, the fork's table of commits, and moving to a new upstream tag. Use when a defect lies in ggml, a ggml kernel should be faster, or the submodule moves (ggml の修正、ggml のフォーク、カーネルの高速化、test-backend-ops、サブモジュールを更新、ggml fork, submodule). Not for a speed-up in speech.cpp's own graphs (speed-up).
---

# Changing the ggml fork

speech.cpp builds ggml from a submodule pinned by commit on the branch `speech.cpp` of github.com/uchkw/ggml: an
upstream release tag, one commit of `test-backend-ops` cases, then one commit for each fix and one for each speed-up,
and nothing else. A defect of ggml is fixed there with cases that show it, rather than avoided in speech.cpp. The
decision record about the fork in `docs/adr/` lists each commit with what shows it and when it is dropped; read it
first.

The fork's upstream is github.com/nyosegawa/ggml. Push campaign branches only to `uchkw/ggml`; do not push them to the
upstream repository. The parent repository is `uchkw/speech.cpp`.

## 1. The branch

- Work in the `ggml/` submodule of your speech.cpp worktree (`git submodule update --init ggml`), on a topic branch
  from the fork's branch: `git -C ggml fetch origin speech.cpp && git -C ggml switch -c <topic> FETCH_HEAD`.
- A fix or a speed-up is one commit on top of the branch, its cases in the same commit, with a message as ggml writes
  them: `<area> : <what it does>` (`vulkan : read ROPE's positions and frequency factors at their offset in a view`).
- Two changes to the fork at once stack: the second branches from the first's topic branch, and its speech.cpp pull
  request stacks on the first's (pull-request skill).

## 2. Cases first

- Add the cases to `tests/test-backend-ops.cpp` before the change: for a fix, the shapes, types and views that show the
  defect at the sizes speech.cpp's models reach; for a speed-up, every layout and shape the faster path takes. The
  cases compare each backend with the CPU.
- Show that a fix's cases fail without it, on the device concerned, and pass with it:
  - **Metal**, on the Mac: `cmake -S ggml -B build-ggml-tests -DGGML_METAL=ON -DGGML_BUILD_TESTS=ON`,
    `cmake --build build-ggml-tests -j4 --target test-backend-ops`, then
    `build-ggml-tests/bin/test-backend-ops -o <OP> -b MTL0`.
  - **Vulkan**, on the Windows machine, which has no compiler, so CI builds the cases: push a temporary fork branch
    with the cases alone, and a temporary speech.cpp branch whose submodule points at it; run
    `gh workflow run build.yml --ref <speech.cpp branch> -f checks=true`; take the artifact
    `speech-checks-windows-x64-vulkan` from that run to the machine (windows-check) and run
    `test-backend-ops -o <OP> -b Vulkan0`. Then delete both temporary branches (`git push origin --delete <branch>` in
    each repository).
- A speed-up's cases pass on Metal and Vulkan with it.

## 3. A speed-up's numbers

Time the model the speed-up is for with the previous submodule's build and the new one in turn, on Metal and on the
Windows machine's Vulkan GPU, as the speed-up skill measures. A speed-up not measured on one of speech.cpp's models
does not go in. The commit message gives the medians with the model, the type, the device and the date.

## 4. Moving the submodule

- Push the topic branch to the fork (`git -C ggml push origin <topic>`) before the speech.cpp branch: CI fetches the
  submodule's commit from the fork, and fails on a commit it cannot reach.
- The speech.cpp pull request moves the submodule (`git add ggml`) and, in the same commit, adds the commit's row to the
  decision record's table (what shows it, when it is dropped) with the measured facts its context needs. It runs the
  stage checks of every model whose graphs use the operation, on the CPU, Metal and Vulkan.
- After the pull request merges, fast-forward the fork's branch to the commit main pins and delete the topic branch:
  `git -C ggml push origin <commit>:refs/heads/speech.cpp`, then `git -C ggml push origin --delete <topic>`. A push
  that is not a fast-forward means another change moved the branch first; rebase onto it, never force.

## 5. A new upstream tag

Moving to a newer tag rebuilds the branch in the decision record's order:

1. The tag, then one commit with the cases of every fix, run on the tag on the devices the table names. A fix is
   dropped, with its row, only when the tag meets the whole of its row's "dropped when", which can ask for more than
   the cases: the row of Metal's matrix units also asks that the tag's kernel give the simdgroup kernel's products up
   to the order of the float sums and be no slower.
2. Each fix the table still calls for, then each speed-up the tag still lacks, with its cases, the latter measured
   again on the new tag.
3. Releases of speech.cpp pin commits of the old branch, and a rebuilt history does not fast-forward from it. After
   the pull request that moves the submodule merges, keep the old branch under a name of its tag
   (`speech.cpp-<old tag>`), so that every earlier release still checks out its submodule, and only then move
   `speech.cpp` to the new history with `--force-with-lease`. This is the one push to the branch that is not a
   fast-forward.

## A subagent's part

A subagent commits in its worktree's submodule and never pushes either repository; the coordinator pushes the fork's
branch and then speech.cpp's.
