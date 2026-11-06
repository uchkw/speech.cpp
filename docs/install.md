# Install

Build and update the `speech` command and `libspeech` from this fork's source.

[uchkw/speech.cpp](https://github.com/uchkw/speech.cpp) has no published binary releases. Its `install.sh` and
`install.ps1` stop with an error and a link to these instructions. Upstream binaries do not contain this fork's changes.

## Build from source

The build needs Git, CMake 3.20 or later and a C++17 compiler. On macOS, install Apple's command line developer tools.
On Windows, use a developer terminal for Visual Studio with C++ tools installed.

```sh
git clone --recurse-submodules https://github.com/uchkw/speech.cpp.git
cd speech.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

On macOS and with a single-configuration generator on Linux or Windows, it makes `build/speech` (`speech.exe` on
Windows) and the shared library beside it. A Visual Studio generator puts them under `build/Release`. A build is tuned
to the CPU it is built on.

On macOS or Linux, add the build folder to `PATH` for the current shell:

```sh
export PATH="$PWD/build:$PATH"
speech --help
```

On Windows with a Visual Studio generator, in PowerShell:

```powershell
$env:PATH = "$PWD\build\Release;$env:PATH"
speech --help
```

Models are fetched when a command first names them; the catalog and Hugging Face sources remain shared with upstream
([models.md](models.md)).

## Backends

| Backend | Configure with | Requirements |
|---|---|---|
| Metal | `cmake -B build` on macOS | Apple silicon or a supported Metal GPU |
| CPU | `cmake -B build` elsewhere | a C++17 compiler |
| Vulkan | `cmake -B build -DGGML_VULKAN=ON` | the Vulkan SDK to build; the Vulkan loader and GPU driver to run |
| CUDA | `cmake -B build -DGGML_CUDA=ON` | the CUDA toolkit; not checked by this fork |

On Linux, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux) brings the build tools in one archive:

```sh
curl -LO https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
tar -xJf vulkansdk-linux-x86_64-1.4.357.0.tar.xz
source 1.4.357.0/setup-env.sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j4
```

## Shared library

Use the `libspeech` built beside the executable and `include/speech.h` from the same checkout
([c-api.md](c-api.md)). On Windows the library is `speech.dll`, with `speech.lib` for linking. CI packages platform
artifacts, but this fork has no published release archives or binary installation channel.

## Update

From a clean checkout of this fork, update the current branch and its pinned submodule, then rebuild:

```sh
git pull --ff-only
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

`speech update` and `speech update --check` both exit with an `unsupported` error and a link to these instructions.
They do not contact a release server or run an installer. Other commands do not check for releases or display update
notices.

## Remove a build

Remove the build directory and any `PATH` entry you added for it. Model files stay in the cache until you remove them
with `speech rm` ([models.md](models.md#where-models-are-kept)).
