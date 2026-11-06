# AGENTS.md

## Project

speech.cpp runs speech models in C++ on [ggml](https://github.com/ggml-org/ggml), on macOS arm64 with Metal,
Windows x64 with Vulkan and Linux x64 with Vulkan or the CPU alone, as a library with one C API,
`include/speech.h`, for any program that speaks text, recognizes speech or detects it: ASIST's worker, other tools, bindings
and other people's applications. Each model's official
implementation is the reference: every stage of a port is checked against tensors dumped from it.
README.md and `docs/` are the documentation for users; what a developer needs is in the code, this file and
`skills/`, and `docs/adr/` keeps the decisions. Read the relevant implementation and its check before changing behavior.

## Architecture

- `include/speech.h` is the C API and the way into the library for every program and binding: plain C,
  opaque handles, UTF-8 strings, errors as statuses whose category says what failed, with the message from
  `speech_last_error()` and the input at fault from `speech_last_error_option()`, and no C++ exception or type
  crossing it. The API has two versions: a change an existing caller notices raises `SPEECH_API_VERSION_MAJOR`,
  which is the shared library's SOVERSION, and an added function, option or enum value raises
  `SPEECH_API_VERSION_MINOR`.
- In `src/api/`, `speech.cpp`, `info.cpp`, `request.cpp` and `voice.cpp` implement the C API over
  `src/engines/engine.h`, the interface of one family behind it, with one engine per family
  (`src/engines/<family>-engine.cpp`). An engine declares in one table the request options its family takes, with their
  defaults and ranges read from the model file, and turns a request checked against that table into the family's; the
  setters, the model information and its JSON read the table and nothing else. One table in `src/api/speech.cpp` lists
  the families: their task (synthesis, recognition, detection, diarization or enhancement), the layout their reader takes, which names the
  `general.architecture` of their files, their engine and, for a family that takes voice files, how it makes one.
  `speech_model_load()` and `speech_model_info_open()` choose the family from the table, and adding a family is adding
  its line.
- `src/families/<family>/` holds the code of one architecture, whichever weights it is given: `qwen3-tts/`
  runs Qwen3-TTS 0.6B and 1.7B. A family reads its model's one GGUF file, its codec included, and turns text into
  audio, or audio into text, into the regions where someone speaks, into who speaks where or into the same audio
  enhanced; it knows nothing of the C API, the worker protocol or the command line. Its
  `layout.cpp` says what its reader takes: the architecture, the layout version, every key with its type, and the
  tensors the keys call for.
- `src/common/` holds what two families use in the same role. Code moves there when a second family needs
  it, not before, and never as a framework for families that do not exist yet.
- `tools/` holds `speech`, the one executable for users, with a subcommand per program: `tts`,
  `asr`, `vad`, `diarize`, `enhance`, `voice`, `info`, `devices`, `quantize` and `update` in `tools/cli/` (with
  `main.cpp`, which dispatches them), `worker` in
  `tools/worker/` and `serve` in `tools/server/`, and what they share in `tools/common/`: the one parser of every
  command line, which makes a flag of each option of the C API's vocabulary, the JSON reader, and the request options
  read from text or JSON and set through the library's setters, transcription by regions, which `asr --vad` and
  `serve` share, the speakers within a region or an utterance (`speakers.cpp`), which `asr --diarize` and `serve`
  share, the enhancement of a recording passed as it settles or gathered whole for a recognition (`enhancement.cpp`),
  which `enhance`, `asr --enhance`, the worker and `serve` share, the synthesis of a text a sentence at a time for a model whose request speaks a short time, which
  `tts` and `serve` share (`sentences.cpp`), the assembly of utterances from audio that arrives while someone speaks
  (`utterances.cpp`), each enhanced whole before its last reading where asked, which `asr -`, `asr --live` and the
  server's Realtime sessions run, and OpenAI's Realtime server
  events (`realtime-events.cpp`), which they all write. Every subcommand reaches the models only through the
  C API. `tools/catalog/` names models: the catalog of the release (`catalog.json`, which `update_catalog.py` writes
  from Hugging Face and the build compiles in), the cache folder, and the fetching of a named model with the system's
  curl (`curl.cpp` runs it), which `models`, `pull` and `rm` and every subcommand that takes a model share. The library never
  reaches the network. The worker's protocol is JSON Lines, one JSON object per line on stdin and
  stdout, and is the contract of every program that starts it, ASIST among them: stdout carries the protocol and
  nothing else, every log goes to stderr, and every request gets exactly one terminal message. Its `ready` message
  carries the protocol's version, the release and the model's information.
- `checks/` holds one check per ported stage (`*-check.cpp`) that compares the stage with the reference dumps, a
  family's in `checks/<family>/` and those of `src/common/` in `checks/common/`, with the headers they share at the top
  of `checks/`, and, in `checks/api/`, `speech-api-check`, which runs the C API through the shared library with a
  synthesis model, Irodori-TTS's own rules and voice files in `speech-api-irodori.c`, with `transcribe`, with a
  recognition model (`speech-api-recognition.c`), with `detect`, with a detection model (`speech-api-detection.c`,
  and detections of audio given a piece at a time in `speech-api-detection-stream.c`), with `diarize`, with a
  diarization model (`speech-api-diarization.c`), and with `enhance`, with an enhancement model
  (`speech-api-enhancement.c`), what they share in
  `speech-api-common.c` and what differs by the operating system in `speech-api-platform.c`. Checks reach into `src/`
  for the stage they check; they are built but not released. `checks/run.py` runs the stage checks of the families and
  devices it is given, with the model files and the dumps it finds, in the order data flows, one line per check.
  `checks/compare/` holds the scripts that compare two builds or releases: the audio of the same requests bit for bit
  (`same_audio.py`), the files `speech quantize` makes with the converters' (`quantize_compare.py`) and the first
  release that reads each (`quantize_releases.py`).
- `measure/` holds what measures and passes or fails nothing: the times of the stages of Qwen3-ASR, FastConformer,
  Qwen3-TTS, Irodori-TTS and SEMamba (`<family>-timing.cpp`, built with the checks), of llama.cpp's llama-server on Qwen3-ASR's
  dumps and of Irodori-TTS with and without a caption, the accuracy of each longest region of transcription by regions
  (`region_cap_compare.py`), and how late live transcription over `/v1/realtime` is for one build or two on the same
  recordings (`live_latency.py`). speech.cpp keeps what passes or fails (the checks, the smoke scripts and
  `checks/compare/`) and what measures from inside the library or its entry points: the time of each stage and the
  delay of live transcription. speech-bench (github.com/nyosegawa/speech-bench) measures what a user sees of a whole
  request (speed, latency, memory and accuracy) across speech.cpp's builds and other runtimes, and tries a model in its
  official implementation and the runtimes that already run it before speech.cpp takes it up, so any other comparison
  of builds or of runtimes goes there and not into `measure/`.
- `tools/server/` holds `speech serve`, which serves a synthesis, a recognition, a detection, a diarization and an
  enhancement model over HTTP with OpenAI's audio API (`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`,
  `GET /v1/models`, `GET /health`), speech.cpp's own `POST /v1/audio/enhancements`, and OpenAI's Realtime transcription
  over a WebSocket at `/v1/realtime`, for programs that speak HTTP. Like the worker it reaches the models only through
  the C API; `openai-api.cpp` reads OpenAI's requests, and the enhancement request in a transcription's form, and writes
  its stream events, `audio-speech.cpp` serves `/v1/audio/speech` and `audio-enhancements.cpp` `/v1/audio/enhancements`,
  both streaming their audio through `audio-stream.cpp`,
  `tools/common/openai-error.cpp` OpenAI's error object, mapped from the library's categories alone,
  `tools/common/realtime-session.cpp` a Realtime session, with commits or `server_vad`, whose messages `realtime.cpp`
  carries over the WebSocket, giving it the recognition, detection, diarization and enhancement models held,
  `jobs.h` runs one request at a time in arrival order and cancels the request of a client that goes away, and
  `served-models.cpp` holds one model of each task and replaces it once its requests have ended. `page/` is the page
  of `speech serve --open`, plain HTML, CSS and JavaScript modules compiled into `speech`, which `page.cpp` serves
  with the endpoints that fetch and load models; `access.cpp` guards them with the Host and the Origin.
- The smoke scripts in `checks/smoke/` drive each entry point as its caller does and fail on a defect:
  `worker_smoke.py`, `worker_recognition_smoke.py` and `worker_enhancement_smoke.py` the worker protocol through
  `worker_client.py`, which checks every line and one terminal message per request, `server_smoke.py` every HTTP
  endpoint and the mapping of errors, `server_realtime_smoke.py` the Realtime sessions of `/v1/realtime`,
  `server_diarization_smoke.py` the speakers of `/v1/realtime` and of `diarized_json`, `server_enhancement_smoke.py`
  `/v1/audio/enhancements` and `noise_reduction` in transcriptions and sessions, and `server_page_smoke.py` the page's
  endpoints and every guard, all through
  `server_client.py`, which stops the
  server however the script ends, `server_page_browser_smoke.mjs` the page itself in a headless Chrome, as a person uses
  it, `server_openai_smoke.py` the server through the official openai-python client,
  `speech_cli_smoke.py` the command line against the worker, and `models_smoke.py` the naming, fetching and removing of
  models against a temporary model folder. A script that needs packages outside the standard library is a uv script
  with its dependencies pinned in the script and in a lock file beside it.
- `reference/<model>/` holds, per model, a uv environment that pins the official code, PyTorch and the rest,
  `pins.py`, which pins the checkpoints by revision, the conversion of the official weights to one GGUF file per
  model, and the scripts that run the official implementation to dump reference tensors. Dumps go to
  `reference/<model>/out/`. `reference/resample/` does the same for the resampler of `src/common/`, against
  torchaudio. `reference/npy_to_wav.py` writes the audio of a dump as a WAVE file.

Keep these boundaries explicit: code does not reach past its module for an operation that belongs to
another one.

## Code

- C++17 with ggml as a git submodule and no other dependency, except two single headers vendored by commit and compiled
  into `speech` alone: cpp-httplib's in `vendor/cpp-httplib/` for `speech serve`, and miniaudio's in `vendor/miniaudio/`
  for the microphone of `speech asr --live`, which loads the system's audio library only when the microphone opens. The
  library and `libspeech` never include either. `speech` has the library and ggml linked in statically, so that a release is
  one executable beside the shared library `libspeech`, which exports the C API and nothing else, for bindings and
  other programs.
- The submodule is the branch `speech.cpp` of github.com/uchkw/ggml: an upstream tag of ggml-org/ggml, a commit of
  `test-backend-ops` cases, one commit for each defect of the tag that speech.cpp's models meet, which its cases
  show, and one commit for each speed-up of speech.cpp's models that the tag lacks, with cases that show it correct
  and its effect measured on Metal and on Vulkan. A defect of ggml is fixed there, with its cases, rather than avoided
  in speech.cpp.
- Anything the C API returns is owned by the library, and the header says for how long. A model serves one
  request at a time, and the header says which functions any thread may call.
- The GGUF layout is this repository's own: one file per model, its codec included, written by
  `reference/<model>/convert.py` and read by the family's `layout.cpp`, with `speech.layout` naming its version.
  The model's identity and languages are in the GGUF specification's own `general.` keys (ggml's `docs/gguf.md`),
  under the names it gives them, and the converter names the file from them by its naming convention; no key of
  speech.cpp's own holds a fact the specification has a key for. Every key is required, but for `general.finetune`
  and `general.version` where a model has none and `general.quantization_version` where it is not quantized, and has
  one type, and the tensors are exactly the ones the keys call for. A change of layout changes the converter and the
  reader in the same commit, and a change a reader of the previous layout cannot read raises `speech.layout`, adds the
  release that reads it to the converter's table and brings the previous layout up in the family's one upgrade
  function. Every model constant lives in the file, from the checkpoint or, where it has none, from the official code,
  and the converter says where.
- Do not add fallback behavior; fail loudly rather than degrade silently. A GGUF without a key or tensor, a key of
  another type, a tensor the keys do not call for, a layout the reader does not know, a text longer than the model
  takes, a WAVE format that is not understood and a device that does not start all throw an `Error` of their kind with a
  message (`src/common/error.h`), naming the input at fault where there is one; the C API returns each with its
  category, which every subcommand passes on: the worker as `error` or `fatal`, the server as OpenAI's error by the
  category alone, and the command line as `speech: <code> (<option>)`. Nothing is truncated or moved to another device
  behind the caller's back: a part of a model's work that runs on another device by design is on the model's page, with
  a request option that keeps it on the model's device. The library checks what it is given before ggml sees it, so that
  no input makes ggml abort the process.
- Fix a defect where its cause is, in a form in which it cannot happen, rather than with a guard for the
  one case that showed it; the code after the fix reads better than before. When a fix is much larger
  than the defect, or needs a choice only the user can make, stop and ask instead of patching.
- Persist each fact in one authoritative place and derive the rest from it.
- Every download is pinned: a Hugging Face repository by revision, a git dependency by commit, Python
  packages by `uv.lock`, the Vulkan SDK in CI by version.
- Extract code only when it creates a coherent responsibility, a reusable boundary or an independently
  checkable unit. Introduce a shared abstraction only after two current implementations show the same
  responsibility with meaningful variation.
- A file approaching 500 lines calls for a review of its responsibilities; split it when a coherent one
  can be extracted, not to meet a line count.
- On Windows, `speech` reads its command line as UTF-8, sets stdin and stdout to binary, and defines
  `NOMINMAX` before `windows.h`.
- A path is a UTF-8 string from the command line to the file. A C stream opens it with `ggml_fopen()` and a
  C++ stream through `std::filesystem::u8path()`: `fopen()` and a stream opened on a `std::string` read the
  path in the ANSI code page on Windows, so a path with any character outside ASCII is not found.
- A Linux release runs on glibc 2.34 and needs no shared library but glibc's and, in the Vulkan build,
  `libvulkan.so.1`; CI fails a build that needs more. It is built on the oldest Ubuntu GitHub
  hosts, with libstdc++ linked in and ggml's OpenMP off.
- A Windows release needs no DLL but Windows' own and `vulkan-1.dll`; CI fails a build that needs more. It is built
  with the C runtime linked in and ggml's OpenMP off.
- Model weights, reference dumps and audio are never committed; `.gitignore` covers `models/`,
  `reference/*/out/`, `*.gguf` and `*.wav`.

## Comments

- Write comments in English, in full sentences and the present tense. Other languages appear only as
  data, quoted verbatim.
- Say only what the code cannot: the layout of a tensor, a quirk of ggml or of the official implementation
  and the failure it causes, a measured value with its condition and date, a constraint, or why the
  simpler approach was rejected. Do not restate names, narrate steps, or mention history, TODOs, docs,
  tickets or conversations.
- Doc comments (`/** */`) on the declarations in headers and on non-obvious declarations in a file,
  without parameter or return tags. A tool starts with a comment that says what it does and its usage.
- Inside a function, a comment goes on its own line above the code. No trailing comments and no banner
  comments.

## Tests

- The checks are the tests. A check runs one stage on the reference dump's inputs, compares its output
  with the dump, prints the error (SNR, largest difference, argmax agreement), and exits with a failure
  when the error is beyond what the stage's arithmetic explains.
- A failing check points to a defect. A check does not fail on a deliberate change of wording, a name or a
  configured value, and does not only assert that something exists.
- Add or update the check of a stage whose behavior changes, and run it on the CPU in F32 and on the GPU
  before committing. A change to code that runs on the GPU is also run on Vulkan, on the Windows machine.
- A port of a model checks each stage in the order data flows, with the stage's own inputs from the dump,
  so that the first stage that departs is the one reported.
- Checks need weights and dumps that are not in the repository, so CI builds them but does not run them.

## Text

- The output of the tools, README.md, `docs/` and AGENTS.md are English. Japanese and other languages appear only
  as data: texts to speak and model output. The one exception is README.ja.md, README.md in Japanese, which says what
  README.md says; a pull request that changes README.md changes README.ja.md the same way.
- An error message names what failed and what to do.
- Speed is RTFx, the seconds of audio over the seconds they took, so that larger is faster, as Hugging Face's Open ASR
  Leaderboard gives it. The other figures take their common names: the time to first audio of a synthesis, the load
  time, the peak memory, and latencies as p50, p90 and p99. The tools' output and the documentation say them alike.

## Decisions

`docs/adr/` keeps the decisions the code cannot show, one file each, named by what it decides and without a number;
the file names are the index. Before changing a behavior, list the folder and read the records whose names cover
it; if the change contradicts one, say so to the user first. Before committing, ask whether the work
settled a choice or turned an approach down for good; if so, the record goes into the same commit (`adr` skill). A
record states the decision as it stands, and nothing else names a record: not the docs, this file, a skill or a
comment.

## Workflow

- Build with `cmake -B build && cmake --build build -j`, and run the checks the change touches before
  committing code, a family's stage checks with `python3 checks/run.py <family> --devices cpu,gpu`. A change to the C
  API or the worker also runs `speech-api-check` and
  `checks/smoke/worker_smoke.py` for each synthesis family, and `speech-api-check transcribe` and
  `checks/smoke/worker_recognition_smoke.py` for both recognition families, `speech-api-check detect` with Silero
  VAD, and `speech-api-check enhance` and
  `checks/smoke/worker_enhancement_smoke.py` with RE-USE; a change to the server runs `checks/smoke/server_smoke.py`,
  `checks/smoke/server_realtime_smoke.py`, `checks/smoke/server_diarization_smoke.py`,
  `checks/smoke/server_enhancement_smoke.py` and `checks/smoke/server_page_smoke.py`, one to the page `checks/smoke/server_page_browser_smoke.mjs`, one to the command
  line or the parser `checks/smoke/speech_cli_smoke.py`, with a model of each family, and one to `tools/catalog/`
  `checks/smoke/models_smoke.py`.
- Never commit on main. Every change reaches main through a pull request, one coherent unit each: a
  model's stage, a fix, a refactor or a documentation change.
- Commit messages and pull request titles are one English sentence in the imperative, without a prefix
  such as `feat:`; the body says what changed, why, and how it was checked.
- The user merges pull requests, with a squash, once CI passes. An agent merges only when told to.
- A defect or a small fix found during other work is fixed in the next pull request that touches its place, or in a
  small one of its own; it is not left for later.
- The worker's protocol has a version of its own, in `ready`, raised when a caller must change to keep working.
- Releasing, the version, the tags, the archives and the files on Hugging Face follow the `release` skill. Converted
  files go to Hugging Face only with the user's approval, and releases and tags are never deleted or moved: callers
  such as ASIST pin them by SHA-256.
- When a change alters what a user does or sees (the C API, the command line's arguments, the worker protocol, the
  HTTP server, the GGUF layout, the models), the same pull request updates the pages that describe it (`docs` skill).
  No test checks the documentation against the code.

## Skills

Skills live in `skills/`; `.claude/skills` and `.agents/skills` link to it. Whenever a task matches one, use it; each
holds steps these rules do not repeat.

- `add-model`: porting a model or a family, updating one, or exposing an official option of one.
- `adr`: a decision settled or an approach turned down for good, a change that contradicts a record, or tidying
  `docs/adr/`.
- `docs`: a change that alters what a user does or sees, or writing a page of README.md or `docs/`.
- `ggml-fork`: a fix or a speed-up in the ggml fork, its `test-backend-ops` cases, and moving the submodule.
- `pull-request`: starting a change, committing, opening a pull request, its review and CI, stacked pull requests,
  merging, cleaning up, and taking back a subagent's branch.
- `release`: model files and cards on Hugging Face, the catalog, the numbers measured again, raising `VERSION`, the
  README's screenshots of the page, the tag and checking the release.
- `speed-up`: making a model faster or smaller in memory, measuring the gain and checking the output.
- `windows-check`: checking a change on a Windows machine with a Vulkan GPU.

## Fork optimization workflow

- `origin` is `uchkw/speech.cpp`; `upstream` is `nyosegawa/speech.cpp`. The ggml submodule follows
  `uchkw/ggml`, with `nyosegawa/ggml` as its upstream. Keep the pinned submodule commit when changing remotes.
- The Haas Mac campaign targets the latency of individual utterances lasting a few seconds to 30 seconds.
  Compare each implementation with the same model file, request options and build flags. A request that explicitly
  selects a weight type keeps that type. Higher weight precision alone is not evidence of better recognition.
- `experiments/haas-mac/` is the narrow exception to the whole-request measurement boundary above: it reproduces this
  fork's per-measure comparisons and renders their evidence. General benchmarking, model evaluation across runtimes,
  and comparisons unrelated to this campaign remain in speech-bench.
- Keep one implementation and its evaluation independently reviewable. An implementation commit and a separate
  report commit may identify the exact source that was measured. Never amend a measured commit or combine unrelated
  measures into one implementation commit.
- Keep compact result JSON and standalone HTML reports in `reports/haas-mac/`. The JSON is the source for the HTML.
  Keep models, audio, reference tensors, builds, raw predictions and logs under ignored `outputs/` directories.
- Keep rejected and inconclusive implementations on experiment branches, with their results and reasons recorded.
  An adopted change is implemented without experiment switches and measured again. Measure cumulative gains against
  a fixed anchor directly; do not add or multiply the gains of individual experiments.
- A report names the source and submodule commits, binary hashes, inputs, options, machine, build flags, warm-up,
  rounds, quality criteria, memory, latency, checks and adoption decision. An unavailable check is unverified, and a
  refused or skipped check is not a pass. The validation requirements above apply unless the maintainer explicitly
  changes the campaign's scope.
