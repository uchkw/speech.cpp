> **Source-only Mac optimization fork:** [uchkw/speech.cpp](https://github.com/uchkw/speech.cpp) has no binary releases. [Build this fork from source](#install) and update its checkout to keep its changes. Binary installers and `speech update` are unsupported.

<p align="center"><img src="docs/images/banner.png" alt="speech.cpp" width="768"></p>

<p align="center">
  Speech synthesis, speech recognition, voice activity detection and speaker diarization for voice conversation, in C++ on ggml
</p>

<p align="center">
  <a href="#models">Models</a> ·
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="docs/c-api.md">C API</a>
</p>

<p align="center">
  English · <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <a href="https://github.com/uchkw/speech.cpp/actions/workflows/build.yml"><img src="https://img.shields.io/github/actions/workflow/status/uchkw/speech.cpp/build.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%C2%B7%20Metal-lightgrey?style=flat-square" alt="macOS on Apple Silicon, with Metal">
  <img src="https://img.shields.io/badge/Windows-x64%20%C2%B7%20Vulkan-lightgrey?style=flat-square" alt="Windows on x64, with Vulkan">
  <img src="https://img.shields.io/badge/Linux-x64%20%C2%B7%20Vulkan%20or%20CPU-lightgrey?style=flat-square" alt="Linux on x64, with Vulkan or the CPU">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"></a>
</p>

speech.cpp is one command, `speech`, and one shared library, `libspeech`, behind one C API, on the CPU, Metal and
Vulkan through [ggml](https://github.com/ggml-org/ggml). The command speaks text, writes the text of recordings, finds
where someone speaks in them, removes their noise, transcribes the microphone as someone speaks (`speech asr --live`),
serves OpenAI's audio API and its Realtime transcription over HTTP, and runs a worker process for any program that keeps models loaded between
requests, [ASIST](https://github.com/nyosegawa/asist) among them.

**A time to first audio of a quarter of a second or less.** On an Apple M5, Qwen3-TTS 0.6B sends its first audio
0.029 s after the text and Irodori-TTS v4.1-Small-MF 0.16 s after it, and the rest follows while it plays.

**One executable, nothing else to install.** No Python and no CUDA: `speech` fetches each model by name, one GGUF file
with its codec, and runs it on Metal, Vulkan or the CPU.

**OpenAI's API on your own computer.** `speech serve` answers OpenAI's audio API and its Realtime transcription, so an
OpenAI client works by changing its base URL.

**Checked against the official implementations.** Every stage of every model is compared with the model's official
implementation, from the same inputs.

<p align="center">
  <img src="docs/images/page-speak.png" alt="The page of speech serve on its Speak tab: a sentence spoken by Qwen3-TTS 0.6B, with its waveform and the time to its first audio" width="720">
  <br>
  <sub>The page of <code>speech serve --open</code>: pick a model of the catalog, fetch it and try it.</sub>
</p>

<table>
  <tr>
    <td width="50%"><img src="docs/images/page-transcribe.png" alt="The Transcribe tab: the same speech written by Qwen3-ASR 0.6B, a region at a time where Silero VAD finds speech"><br><sub>Transcribe writes a file or a recording, a region at a time where someone speaks.</sub></td>
    <td width="50%"><img src="docs/images/page-live.png" alt="The Live tab while someone speaks: the first sentence final and the next one in grey as it is said"><br><sub>Live writes the microphone as you speak; the grey text is the utterance still being said.</sub></td>
  </tr>
</table>

## Models

| Task | Name | Languages | Good at |
|---|---|---|---|
| Synthesis | [`qwen3-tts-0.6b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF), [`qwen3-tts-1.7b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | de en es fr it ja ko pt ru zh | nine named speakers; audio that streams from its first frame of 0.08 s; the 1.7B model follows an instruction of how to speak |
| Synthesis | [`irodori-tts-mf`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF), [`irodori-tts`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF), [`irodori-tts-large`](https://huggingface.co/sakasegawa/Irodori-TTS-v4-Large-GGUF) | ja | Japanese in the voice of a reference recording, or in a voice described in words; the large model follows both more closely |
| Synthesis | [`drama-3`](https://huggingface.co/sakasegawa/drama-3-GGUF) | 83 languages | a scene of up to 16 speakers, with cues such as laughter written in the text, in voices cloned from recordings and their words; a 20.8 GB file. Built with Fish Audio |
| Recognition | [`qwen3-asr-0.6b`](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF), [`qwen3-asr-1.7b`](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF) | 30 languages | finds the language itself, takes a prompt of names and terms, and up to 20 minutes at once |
| Recognition | [`reazonspeech-v2`](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) | ja | Japanese with punctuation, in recordings of many minutes |
| Recognition | [`parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) | ja | short Japanese utterances, fast, with times |
| Recognition | [`parakeet-tdt-0.6b-v3`](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) | 25 European languages | finds the language itself, with punctuation, capitals and times |
| Recognition | [`haas-ja-v1.0`](https://huggingface.co/sakasegawa/Haas-ja-v1.0-GGUF) | ja | Japanese with punctuation, fast, with times or a custom vocabulary |
| Recognition | [`nemotron-3.5-transcribe`](https://huggingface.co/sakasegawa/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF) | 40 languages | word for word as spoken, fillers included, up to an hour at once, with a prompt of names and terms, or with who speaks and each word's time; a 19.0 GB file |
| Detection | [`silero-vad`](https://huggingface.co/sakasegawa/silero-vad-GGUF) | any | where someone speaks, to give a recognizer one utterance at a time; 1.2 MB |
| Diarization | [`nemotron-3-diarization`](https://huggingface.co/sakasegawa/Nemotron-3-Diarization-GGUF) | any | who speaks when, up to eight speakers, in a recording or as it arrives; 106 MB |
| Enhancement | [`re-use`](https://huggingface.co/sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF), [`re-use-laughter`](https://huggingface.co/sakasegawa/NVIDIA-NemotronLabs-RE-USE-laughter-GGUF) | any | removes noise, reverberation and band limits from a recording at 8 to 48 kHz, for people to listen to; the laughter model keeps laughter; 38.5 MB |

A command takes a model by its name and fetches its converted file from Hugging Face, from the repository the name links
to, the first time. [docs/models.md](docs/models.md) lists the models with their sizes, voices and languages, and each
family has a page: [Qwen3-TTS](docs/models/qwen3-tts.md), [Irodori-TTS](docs/models/irodori-tts.md), [Drama 3](docs/models/drama3.md),
[FastConformer](docs/models/fastconformer.md), [Qwen3-ASR](docs/models/qwen3-asr.md),
[Nemotron 3.5 Transcribe](docs/models/nemotron-transcribe.md), [Silero VAD](docs/models/silero-vad.md),
[Sortformer](docs/models/sortformer.md) and [SEMamba](docs/models/semamba.md).

## Install

On macOS, install CMake 3.20 or later and a C++17 compiler, then build this fork with Metal:

```sh
git clone --recurse-submodules https://github.com/uchkw/speech.cpp.git
cd speech.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
export PATH="$PWD/build:$PATH"
speech --help
```

This sets `PATH` for the current shell. [docs/install.md](docs/install.md) covers other platforms, shared libraries
and updating from source. This fork does not check for binary releases or display update notices. Model names and
Hugging Face sources remain the same as upstream.

## Quick start

1. Speak a sentence. The model is fetched the first time (1.21 GB):

   ```sh
   speech tts qwen3-tts-0.6b --voice ryan -o hello.wav "Hello. It will be sunny in Tokyo tomorrow."
   ```

2. Write its text back (0.84 GB the first time):

   ```sh
   speech asr qwen3-asr-0.6b hello.wav
   ```

   ```
   Hello, it will be sunny in Tokyo tomorrow.
   ```

3. Open a page on which to pick models, speak and transcribe:

   ```sh
   speech serve --open
   ```

`speech models` lists every model and, for each language, the model to start with.

## Command line

| Subcommand | What it does |
|---|---|
| `speech tts` | speaks text into a WAVE file or to stdout |
| `speech asr` | writes the text of WAVE files, or of the microphone as someone speaks |
| `speech vad` | writes where someone speaks in WAVE files |
| `speech diarize` | writes who speaks where in WAVE files |
| `speech enhance` | removes noise and other degradations from WAVE files |
| `speech voice` | makes a voice file for Irodori-TTS or Drama 3 from reference recordings |
| `speech info` | prints what a model file says of its model |
| `speech devices` | lists the devices a model can run on |
| `speech models`, `pull`, `rm` | list, fetch and remove models |
| `speech quantize` | writes a model file of F32 weights in F16, Q8_0 or a smaller type |
| `speech update` | explains how to update this source-only fork |
| `speech serve` | serves models over HTTP with OpenAI's audio API, and a page to try them |
| `speech worker` | serves a model over JSON Lines on stdin and stdout, for programs |

```sh
speech voice irodori-tts-mf me.wav me.voice.gguf
speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o - < story.txt | ffplay -nodisp -autoexit -
speech asr reazonspeech-v2 --timestamps meeting.wav
speech asr qwen3-asr-1.7b --language ja --prompt "Claude Code、渋谷" meeting.wav
speech asr nemotron-3.5-transcribe --speakers meeting.wav
speech asr reazonspeech-v2 --vad silero-vad --live
speech vad silero-vad --max-speech-duration-s 10 meeting.wav
speech enhance re-use street.wav -o clean.wav
```

Every request option is a flag, and `speech info MODEL` lists the ones a model takes. [docs/cli.md](docs/cli.md) lists
every subcommand and option.

## Server and page

`speech serve` holds a synthesis model, a recognition model, a detection model and a diarization model, and answers
`POST /v1/audio/speech` and `POST /v1/audio/transcriptions` as OpenAI's audio API does, with the speakers of
`diarized_json`, for a web app, a script or curl:

```sh
speech serve qwen3-tts-0.6b qwen3-asr-0.6b
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "Hello.", "voice": "ryan"}' -o hello.wav
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@hello.wav
```

It listens on 127.0.0.1 and has no authentication. [docs/server.md](docs/server.md) has the endpoints, the formats, the
errors and the page.

## Using the library

A program links `libspeech` and includes `speech.h`, from a release archive:

```c
speech_model * model = NULL;
speech_request * request = NULL;
speech_model_load("Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf", NULL, &model);
speech_request_new(model, &request);
speech_request_set_text(request, "Hello.");
speech_request_set_string(request, SPEECH_OPT_VOICE, "ryan");
speech_synthesize(request, on_audio, NULL);   /* on_audio receives the samples as they are made */
```

Each call returns a status whose category says what failed. [docs/c-api.md](docs/c-api.md) has the whole API.

A program that would rather not link a library starts `speech worker MODEL` and speaks JSON Lines with it, as ASIST
does: one request per line on stdin, its audio or text and one terminal message on stdout.
[docs/worker.md](docs/worker.md) has the protocol.

## Accuracy and speed

Every stage of each port is compared with tensors dumped from the official implementation. The recognizers write the
official text on every check input in F32 and F16.

Measured with [speech-bench](https://github.com/nyosegawa/speech-bench) on an Apple M5 (Metal) on 2026-10-16 and an
RTX 2080 (Vulkan) on 2026-10-14 and 16, one request at a time. Speech: 20 Japanese sentences with seeds 1 and 2, three rounds, Irodori-TTS
in the voice of one reference recording; the p50 time to first audio, the RTFx (the seconds of audio over the seconds
they took, so larger is faster), and the CER of the speech as Qwen3-ASR 1.7B hears it. Drama 3 was measured on
2026-10-16 with seeds 1 to 3, in the voice of one reference recording, with the RTX 2080 computing the experts of 35 of
its 40 layers on the CPU.

| Model | Time to first audio, M5 / RTX 2080 | RTFx | CER |
|---|---|---|---|
| `qwen3-tts-0.6b`, Q8_0 | 0.029 s / 0.024 s | 3.7 / 5.2 | 8.1% / 6.0% |
| `qwen3-tts-1.7b`, Q8_0 | 0.041 s / 0.032 s | 2.6 / 4.2 | 4.5% / 6.1% |
| `irodori-tts-mf`, F16 | 0.16 s / 0.12 s | 9.0 / 18 | 6.1% / 6.4% |
| `irodori-tts`, F16, 16 steps | 0.75 s / 0.52 s | 4.9 / 8.5 | 3.2% / 3.2% |
| `irodori-tts-large`, Q8_0, 16 steps | 2.2 s / 1.25 s | 1.7 / 3.3 | 5.4% / 5.5% |
| `drama-3`, Q4_K | 0.28 s / 0.58 s | 1.1 / 0.57 | 1.8% / 2.4% |

Recognition: the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, with the CER that
accepts other spellings of the same words in parentheses, the same on both machines, and the p50 latency over the same
utterances; parakeet-tdt-0.6b-v3 on 300 English utterances of FLEURS, three rounds.

| Model | CER on Common Voice ja | p50 latency, M5 / RTX 2080 |
|---|---|---|
| `parakeet-tdt_ctc-0.6b-ja`, Q8_0 | 7.9% (3.0%) | 0.025 s / 0.047 s |
| `reazonspeech-v2`, Q8_0 | 12.0% (7.1%) | 0.049 s / 0.096 s |
| `qwen3-asr-0.6b`, Q8_0 | 11.9% (7.0%) | 0.107 s / 0.072 s |
| `qwen3-asr-1.7b`, Q8_0 | 9.5% (4.7%) | 0.257 s / 0.140 s |
| `parakeet-tdt-0.6b-v3`, F16 | WER 8.8% on FLEURS en | 0.050 s / 0.083 s |
| `haas-ja-v1.0`, Q8_0 | 4.4% (2.9%) on FLEURS ja | 0.042 s / 0.097 s |
| `nemotron-3.5-transcribe`, IQ4_NL | 9.2% (5.3%) | 0.61 s / - |

The FastConformer models (parakeet, ReazonSpeech and Haas-ja-v1.0) were measured again on 2026-10-31 with 0.9.4's
build. Nemotron 3.5 Transcribe was measured on 2026-10-29: its CER on an H100 with Vulkan, and its latency on the M5
over the first 100 utterances. Haas-ja-v1.0 was trained on Common Voice, so it was measured on the 650 utterances of
FLEURS' Japanese test split, where parakeet-tdt_ctc-0.6b-ja gave 5.0% (2.8%) on 2026-10-30. Each model's page has
more.

## Documentation

| Page | What it covers |
|---|---|
| [Install](docs/install.md) | building from source, backends, shared libraries and updating |
| [Command line](docs/cli.md) | every subcommand and option |
| [Models](docs/models.md) | the catalog, names, where models are kept, voices and languages |
| [Qwen3-TTS](docs/models/qwen3-tts.md), [Irodori-TTS](docs/models/irodori-tts.md), [Drama 3](docs/models/drama3.md), [FastConformer](docs/models/fastconformer.md), [Qwen3-ASR](docs/models/qwen3-asr.md), [Nemotron 3.5 Transcribe](docs/models/nemotron-transcribe.md), [Silero VAD](docs/models/silero-vad.md), [Sortformer](docs/models/sortformer.md), [SEMamba](docs/models/semamba.md) | each family: what it does, its options, accuracy and speed |
| [Server and page](docs/server.md) | `speech serve` and OpenAI's audio API |
| [Worker protocol](docs/worker.md) | `speech worker` for programs |
| [C API](docs/c-api.md) | `speech.h` |

## License

MIT, see [LICENSE](LICENSE). The model weights are their authors':

- Qwen3-TTS and Qwen3-ASR are the Qwen team's, under the Apache License 2.0.
- Irodori-TTS v4.1-Small and v4.1-Small-MF are Aratako's, under the MIT License with the ethical restrictions of their
  model cards: no voice cloning without consent, no deepfakes or misinformation.
- Irodori-TTS v4-Large is Aratako's, under the Gemma Terms of Use, since its text encoder derives from Google's
  T5Gemma 2, with the same ethical restrictions.
- Semantic-DACVAE-Japanese-32dim, the codec in the Irodori-TTS files, is Aratako's under the MIT License. It derives
  from Meta's facebook/dacvae-watermarked, under the Apache License 2.0; the SAM License that its card's text also
  names was corrected to Apache-2.0 in facebookresearch/dacvae on 2025-12-19.
- Drama 3 and its codec are Fish Audio's, under the Fish Audio Community License 1.0: research and non-commercial use
  is free, commercial use is allowed below USD 500,000 of yearly revenue, and a speech or voice-cloning service for
  others needs a separate license from Fish Audio. Built with Fish Audio.
- The parakeet models are NVIDIA's, under CC-BY-4.0, reazonspeech-nemo-v2 is reazon-research's and Haas-ja-v1.0 is
  Third Intelligence's, both under the Apache License 2.0.
- Silero VAD is the Silero Team's, under the MIT License.
- Nemotron 3 Diarization and Nemotron 3.5 Transcribe are NVIDIA's, under the OpenMDW License Agreement 1.1.
