# speech fetches models with the system's curl into the system's cache folder, and the library never reaches the network

## Context

A model named from the catalog is fetched over HTTPS from Hugging Face and its content servers, through whatever proxy
and certificates the user's system has: corporate proxies, a certificate a company installs, or `https_proxy` set for
the shell. speech.cpp has no dependency but ggml and cpp-httplib's header, which speaks HTTPS only when linked with
OpenSSL. macOS, Windows 10 1803 and later, and nearly every Linux distribution ship curl.

A model file is 0.8 to 8 GB, several programs may fetch the same one at once (a script running `speech asr` while
another program starts `speech serve` or a worker), and a fetch over a slow line can stop halfway. Each release's
catalog pins its own revisions, so after an upgrade the folder can hold files the new release names no more, which an
older release still installed may use.

## Decision

- **`speech` runs the system's curl**: `curl` on the PATH on macOS and Linux, and `System32\curl.exe` on Windows, which
  a bare `curl.exe` would look for in the current folder first. curl follows the redirect to Hugging Face's content
  servers, honors the proxy variables and its own configuration, and checks certificates against the system's store
  (Schannel on Windows, the keychain for macOS's curl). Without curl, a fetch fails saying to install it or to give a
  model file's path.
- **curl writes the file's bytes to a pipe**, and `speech` appends them to `<file>.part`, hashing them as they come, so
  that no path, which may hold characters outside ASCII, crosses to a program that may read it in another encoding. curl
  gets no handle or file descriptor of `speech` but its stdin, stdout and stderr, so it holds no model file or socket of
  the server open. A fetch that stops keeps the part, and the next one asks for the rest with `--continue-at`. The file
  is renamed into place once its size and SHA-256 are the catalog's; a part whose size or SHA-256 cannot be the file's
  is removed, and the failure says that the same command fetches it again. A transfer slower than 1 KiB/s for a minute
  fails rather than hangs. Progress goes to stderr, one line rewritten on a terminal and a line for each tenth
  elsewhere, so that a worker's stdout carries the protocol alone.
- **The folder is the system's cache folder**: `~/Library/Caches/speech.cpp/models` on macOS,
  `$XDG_CACHE_HOME/speech.cpp/models` or `~/.cache/speech.cpp/models` on Linux (the XDG Base Directory specification,
  which ignores a value that is not absolute), and `%LOCALAPPDATA%\speech.cpp\models` on Windows. `SPEECH_MODEL_DIR`
  replaces it and must be an absolute path, so that its meaning does not depend on the current folder.
- **Inside it, a file goes to `<owner>--<name>/<sha256>/<file>`**, the repository and the SHA-256 of the file's bytes,
  so that a path, once filled, always holds the same bytes, and a release whose catalog pins the repository at a later
  commit holding the same file finds it in place. The commit a catalog pins only makes the URL to fetch from, as
  Hugging Face's own cache, llama.cpp and Ollama tell files by their content. A file is there only once it is whole, so
  a name whose file is there loads without the network.
- **Two processes that need the same file take turns on `<file>.lock`**, `flock()` on macOS and Linux and `LockFileEx()`
  on Windows, which the system releases when a process ends however it ends; the second finds the file in place and
  loads it. The lock is held while a process fetches or removes the file.
- **Nothing is removed automatically.** `speech models` shows every file in those folders that no entry of this
  release's catalog names as old, with its size, and `speech rm --old` removes them; `speech rm NAME[:TYPE]` removes a
  model's file and what was fetched of it. Only the folders a fetch makes are looked into, so a `SPEECH_MODEL_DIR` that
  also holds other files never offers them for removal.
- **Only the command line and the server fetch.** The library and the C API load files and never reach the network; a
  binding or a program on `libspeech` fetches as it chooses, and can read the pins from `speech models --json`.

The alternatives were turned down:

- cpp-httplib with OpenSSL. OpenSSL would be a second dependency in every release, with a certificate store of its own
  that a company's certificate and the system's settings do not reach.
- Each system's own library (NSURLSession, WinHTTP, libcurl on Linux). Three implementations of the same thing, and
  linking libcurl would add a shared library the Linux releases do not carry.
- curl writing the part itself with `--output`. A path outside ASCII on Windows depends on how the installed curl reads
  its command line, and `speech` could not hash the bytes as they arrive.
- Fetching inside the library. Programs that pin and fetch their files themselves would carry network code they never
  call, and a library that may start downloads is harder to trust.
- Removing a release's old files when a newer release first runs. Another release still installed, or a program that
  pinned that release, may be using them, and a download of gigabytes is not to be thrown away without being asked.
- A folder of its own beside the executable or under `~/.speech.cpp`. A cache folder is where each system keeps files
  that can be fetched again, and where its users and their cleaning tools look for them; beside the executable, an
  update of speech.cpp would have to carry the models along.
- Hugging Face's own cache (`~/.cache/huggingface/hub`). Its layout is the hub library's to change, and its blobs and
  symbolic links work differently on Windows without Developer Mode.
- A flat folder of file names. Two contents of a file of the same name would overwrite each other.
- A folder for each commit a catalog pins, `<owner>--<name>/<revision>/<file>`. A commit that changes only the model
  card, or adds a file of another type, moves the pin, and every file of the repository is fetched again though its
  bytes did not change: the speeds added to four cards after 0.7.1 moved the pins of Qwen3-ASR 0.6B and 1.7B and
  Qwen3-TTS 0.6B and 1.7B, 6.5 GB of files that had not changed.
- Pinning each model at the newest commit that changed one of its files, with a folder per commit. A file of another
  type added to the repository, such as a lower-bit one, would still move the pin of the files beside it.

## Consequences

A fetch behaves as the user's curl does, proxies and certificates included. A machine without curl, such as a minimal
container, fetches nothing and takes paths. Resuming depends on the server answering ranges, which Hugging Face's
content servers do. A file fetched once serves every program that names the same bytes, at whatever commit. An upgrade
leaves the previous release's files until the user removes them, which `speech models` points out. Uninstalling
speech.cpp leaves the folder, which docs/install.md says how to remove. The `<revision>` folders of releases before
0.8.2 are neither moved nor listed as old.
