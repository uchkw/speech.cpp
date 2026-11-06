# Maintain the Haas Mac optimization fork

This repository maintains measured Haas-ja-v1.0 optimizations for Apple Silicon, with one implementation and its
evaluation recorded at a time. It preserves the speech C API, explicit model selection and upstream history.

| Repository | Fork | Upstream |
|---|---|---|
| speech.cpp | https://github.com/uchkw/speech.cpp | https://github.com/nyosegawa/speech.cpp |
| ggml | https://github.com/uchkw/ggml, branch `speech.cpp` | https://github.com/nyosegawa/ggml, branch `speech.cpp` |

## Build and update from source

This fork has no published binary releases. Build the fork checkout and its pinned ggml submodule; upstream binaries
do not contain this fork's changes. Do not use the upstream installers or binary update commands for this checkout.
On macOS with a C++17 compiler and CMake 3.20 or later:

```sh
git clone --recurse-submodules https://github.com/uchkw/speech.cpp.git
cd speech.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
./build/speech --help
```

To update a clean checkout, run `git pull --ff-only` and `git submodule update --init --recursive`, then configure and
build again. CI builds and packages the platform artifacts, but its release-installer job is explicitly skipped
because this fork has no release channel. This is unperformed installation/update validation, not a passing check.
Model catalog entries and Hugging Face sources remain shared with upstream.

## Maintain the experiments

The campaign is tracked in [issue 1](https://github.com/uchkw/speech.cpp/issues/1). Small JSON summaries and standalone
HTML reports live in `reports/haas-mac/`; their reproducible runner and renderer live in `experiments/haas-mac/`.
Large inputs, raw results and builds stay in ignored `outputs/` directories. Open an HTML report locally to view it.

Use separate worktrees and topic branches. Keep the initial source baseline and each measured implementation as
immutable commits. Commit the evaluation report separately when its evidence names the implementation commit. Keep
failed experiments on their own branches, and bring measured improvements into the maintained branch through pull
requests. The maintainer merges them.

To update from upstream, fetch `upstream`, create an integration branch from the maintained fork branch, and review
the changes before merging upstream. Keep the ggml commit pinned until its fixes and optimization results have been
checked against the proposed update. Rebuild and rerun the affected comparisons; earlier measurements do not establish
the performance of a new upstream commit.

The first campaign anchor is speech.cpp `03555100287c118b41d1b475044e5ab1e07baa8c` with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. Source-derived build manifests identify the actual anchor binaries used.
No fork release or model publication is implied by an experiment result.
