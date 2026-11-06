# FastConformer

This page describes speech.cpp's port of NVIDIA NeMo's [FastConformer](https://arxiv.org/abs/2305.05084) recognizers:
two parakeet models, ReazonSpeech and Haas, which write the text of a recording.

| Name | Upstream | Languages | File |
|---|---|---|---|
| `reazonspeech-v2` | [reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | ja, with punctuation | `reazonspeech-nemo-619M-v2-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt_ctc-0.6b-ja` | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | ja | `parakeet-tdt_ctc-0.6B-ja-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt-0.6b-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 European languages, with punctuation and capitals | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB |
| `haas-ja-v1.0` | [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | ja, with punctuation | `Haas-616M-ja-v1.0-Q8_0.gguf`, 0.66 GB |

parakeet-tdt-0.6b-v3's languages are those of its model card: `bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hr`
`hu` `it` `lt` `lv` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `uk`.

```sh
speech asr parakeet-tdt_ctc-0.6b-ja utterance.wav                  # the text on stdout
speech asr haas-ja-v1.0 --timestamps utterance.wav                 # Japanese with punctuation, with times
speech asr haas-ja-v1.0 --keywords $'Haas=ハース\nAIモデル' talk.wav    # with a custom vocabulary of two keywords
speech asr parakeet-tdt-0.6b-v3 --timestamps utterance.wav         # with the times of its segments
speech asr reazonspeech-v2 --vad silero-vad meeting.wav            # a recording of minutes, by its regions of speech
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # faster, with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## What it does

- Each model writes the text NeMo's `transcribe()` writes by default: ReazonSpeech with a beam search, the parakeet models
  and Haas with greedy decoding. Haas writes a rare character its vocabulary lacks from its UTF-8 bytes, as its
  tokenizer does.
- A request is recognized whole, however long, but a stretch that holds several sentences loses whole sentences, as in
  NeMo itself: on minutes of Common Voice ja clips joined with pauses, reazonspeech-v2 dropped 522 of 600 and
  parakeet-ja 266. Recognize a long recording by the regions where someone speaks, with `--vad silero-vad` or
  `chunking_strategy`. ReazonSpeech's time and memory grow with the length; the parakeet models attend over the whole
  request, so theirs grow with its square, and so do Haas's.
- parakeet-v3 finds the language of the audio itself. None of the models takes a language: a request's `language` is only
  checked against the model's languages.
- With `timestamps`, the result also has the tokens and segments with their times in seconds. A segment ends at a full
  stop, a question mark or an exclamation mark.
- On a GPU, the encoder runs on the GPU, and the decoding runs on the CPU where the file's decoding weights are
  quantized (Q8_0, Q6_K, Q5_K and Q4_K files) and on the GPU where they are F16 or F32: each step of the decoding is a
  small computation, which the CPU finishes before a round trip to the GPU would with quantized weights but not with F16
  ones. `decode_on_cpu` chooses either way per request. When a request first decodes on the CPU, the model copies the
  weights the decoding reads into the CPU's memory and keeps them, 12 MB for the Q8_0 files of reazonspeech-v2 and
  parakeet-ja and 35 MB for the F16 file of parakeet-v3, and the CPU computes with `--threads`. The CPU's arithmetic
  differs from the GPU's, so a long recording can come out a word different where two words score almost the same.

Not supported: the CTC head of parakeet-tdt_ctc-0.6b-ja, which NeMo does not decode with by default, and
`transcribe_longform()` of Haas's Python package; recognize a long recording by its regions instead.

## A custom vocabulary (Haas)

Haas takes the custom vocabulary of its Python package, Haas-ASR: names and terms it rarely heard in training, which it
then recognizes more often and writes as given. `keywords` lists them, one a line, each its written form followed by "="
and the readings Haas would write for it, usually in kana (`Haas=ハース,ハウス`); a keyword Haas already writes as listed
needs none (`AIモデル`). The decoding boosts each keyword's readings and written form, the text writes each keyword found
in it in its written form whether in hiragana or katakana, full-width or not, in any case or spacing, and the result is
Haas-ASR's text for the same words. Over HTTP, OpenAI's `keywords` array carries one keyword with its readings in each
string ([server.md](../server.md)).

Boosting can also change text where no listed word was said. As Haas-ASR does, audio without speech is not boosted, and
each request is also decoded without the keywords: a change is kept only where it holds a whole listed keyword
(`keywords_guard`). Both decodings share one pass of the encoder. List only the keywords you need: each adds a small
chance that a word is written where it was not said.

A recognition with keywords gives its text alone, since the keywords write it after the decoding, so `timestamps` with
keywords is refused. Files of Haas from before speech.cpp 0.9.4 hold no vocabulary; fetch the file again
(`speech pull haas-ja-v1.0`) to use one.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the model's languages; only checked |
| `timestamps` | false | add the times of the tokens and segments |
| `decoding` | `beam` | reazonspeech-v2 only: `beam` or `greedy`. Greedy is faster and can write another text, often with fewer commas and full stops |
| `decode_on_cpu` | true for Q8_0 and K-quant files, false for F16 and F32 | for a model on a GPU, decode on the CPU (true) or on the GPU (false). A model on the CPU decodes there either way |
| `keywords` | `""` | Haas only: the custom vocabulary, one keyword a line, `written form=reading,reading`; not with `timestamps` |
| `keywords_alpha` | 2 | Haas only: how strongly the keywords are boosted; larger fixes more missed words and writes more words where they were not said |
| `keywords_guard` | true | Haas only: keep a change the keywords made only where it holds a whole keyword |

## Accuracy

Every stage is checked against NeMo's tensors on utterances of FLEURS' test split, on an Apple M5:

| Model | Inputs | Text from the audio, CPU F32, CPU F16 and Metal |
|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 3 ja_jp utterances, 6 to 26 s | NeMo's text on all three |
| parakeet-tdt-0.6b-v3 | 12 utterances of en_us, de_de, fr_fr and es_419, 6 to 23 s | NeMo's text on all twelve |
| reazonspeech-nemo-v2 | 8 ja_jp utterances and 2 joined inputs of 65 s and 311 s | NeMo's text on all ten, with beam search and with greedy decoding |
| Haas-ja-v1.0 | 5 ja_jp utterances, 6 to 26 s, two of them with characters it writes from their bytes | NeMo's text on all five |

The times of tokens and segments are NeMo's exactly with F32 weights, and with F16 weights decoding on Metal. Decoding on
the CPU with F16 weights, a token can fall one frame (80 ms) from NeMo's where two frames score almost the same: one
token in 2 of reazonspeech-nemo-v2's 20 decodings.

On the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, the CER in Q8_0 is 7.93% for
parakeet-tdt_ctc-0.6b-ja, 12.04% for reazonspeech-nemo-v2 with beam search and 12.47% with greedy decoding (3.01%,
7.12% and 7.66% accepting other spellings of the same words), on the M5 and within 0.01 points on the RTX 2080.
parakeet-tdt-0.6b-v3's WER in F16 on 300 English utterances of FLEURS is 8.76% on the M5 and 8.78% on the RTX 2080.

Haas-ja-v1.0 was trained on Common Voice 26.0, and 1,033 of that test set's clips are in Common Voice 17.0's training
split, so it is compared on the 650 utterances of FLEURS' Japanese test split, which its card says it was not trained
on. In Q8_0 on the M5 (2026-10-30) its CER there is 4.39%, against 5.00% for parakeet-tdt_ctc-0.6b-ja and 7.95% for
reazonspeech-nemo-v2 with beam search (2.86%, 2.82% and 5.70% accepting other spellings of the same words).

## Weight types

Each model comes in five types; `NAME:TYPE` fetches one (`speech asr parakeet-tdt-0.6b-v3:q8_0`). speech-bench, two
rounds, on an Apple M5 with Metal and an RTX 2080 with Vulkan on 2026-10-31, with the release's build. The error is the
CER on the first 500 utterances of Common Voice 8.0's Japanese test set, on which the CER is higher than on the whole
set, and after the slash the CER accepting other spellings of the same words; for parakeet-tdt-0.6b-v3 it is the WER on
300 English utterances of FLEURS, and for haas-ja-v1.0 the CER on the 650 utterances of FLEURS' Japanese test split.
The error is the M5's, and the RTX 2080's is within 0.05 points of it. The peak memory
is the process's on the M5, whose GPU shares it, and the VRAM the video memory in use on the RTX 2080.

| Model | Type | File | Error | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | F16 | 1.24 GB | 9.09% / 2.37% | 1.28 GiB | 1.23 GiB | 113 / 80 |
| parakeet-tdt_ctc-0.6b-ja | Q8_0 | 0.66 GB | 9.14% / 2.35% | 0.74 GiB | 0.69 GiB | 143 / 74 |
| parakeet-tdt_ctc-0.6b-ja | Q6_K | 0.51 GB | 9.08% / 2.48% | 0.60 GiB | 0.56 GiB | 136 / 80 |
| parakeet-tdt_ctc-0.6b-ja | Q5_K | 0.43 GB | 8.79% / 2.35% | 0.53 GiB | 0.48 GiB | 130 / 79 |
| parakeet-tdt_ctc-0.6b-ja | Q4_K | 0.36 GB | 9.26% / 2.48% | 0.45 GiB | 0.41 GiB | 143 / 73 |
| reazonspeech-nemo-v2, beam search | F16 | 1.24 GB | 14.12% / 7.14% | 1.33 GiB | 1.25 GiB | 50 / 30 |
| reazonspeech-nemo-v2, beam search | Q8_0 | 0.66 GB | 14.17% / 7.15% | 0.79 GiB | 0.70 GiB | 72 / 34 |
| reazonspeech-nemo-v2, beam search | Q6_K | 0.51 GB | 13.91% / 6.98% | 0.65 GiB | 0.57 GiB | 69 / 34 |
| reazonspeech-nemo-v2, beam search | Q5_K | 0.43 GB | 13.96% / 7.06% | 0.58 GiB | 0.51 GiB | 69 / 32 |
| reazonspeech-nemo-v2, beam search | Q4_K | 0.36 GB | 14.20% / 7.10% | 0.50 GiB | 0.42 GiB | 71 / 34 |
| parakeet-tdt-0.6b-v3 | F16 | 1.26 GB | 8.76% | 1.31 GiB | 1.35 GiB | 151 / 97 |
| parakeet-tdt-0.6b-v3 | Q8_0 | 0.67 GB | 9.04% | 0.77 GiB | 0.80 GiB | 195 / 95 |
| parakeet-tdt-0.6b-v3 | Q6_K | 0.52 GB | 8.87% | 0.63 GiB | 0.67 GiB | 181 / 100 |
| parakeet-tdt-0.6b-v3 | Q5_K | 0.44 GB | 9.12% | 0.56 GiB | 0.60 GiB | 177 / 94 |
| parakeet-tdt-0.6b-v3 | Q4_K | 0.36 GB | 9.51% | 0.49 GiB | 0.52 GiB | 193 / 92 |
| haas-ja-v1.0 | F16 | 1.23 GB | 4.40% / 2.87% | 1.29 GiB | 1.32 GiB | 169 / 98 |
| haas-ja-v1.0 | Q8_0 | 0.66 GB | 4.39% / 2.86% | 0.74 GiB | 0.78 GiB | 205 / 98 |
| haas-ja-v1.0 | Q6_K | 0.51 GB | 4.35% / 2.83% | 0.61 GiB | 0.65 GiB | 192 / 99 |
| haas-ja-v1.0 | Q5_K | 0.43 GB | 4.47% / 2.87% | 0.53 GiB | 0.58 GiB | 190 / 96 |
| haas-ja-v1.0 | Q4_K | 0.35 GB | 4.42% / 2.84% | 0.46 GiB | 0.50 GiB | 205 / 90 |

In the Q6_K, Q5_K and Q4_K files the prediction network and the joint stay Q8_0. A quantized file decodes on the CPU
by default and an F16 file on the GPU; `decode_on_cpu` sets where a request decodes.

Haas Q4_K and Q8_0 on Metal automatically keep an extra 40.5 MiB and 76.5 MiB of encoder weights, respectively,
for faster recognition, at the cost of more memory and load time. The memory figures above exclude this copy.
It applies to the standard Haas files; CPU and Vulkan encoders, other weight types and files with mixed attention
weight types keep no such copy. Its speed benefit has only been measured on an M2 Ultra, not on other Metal devices.

## Speed

speech-bench, on the M5 and the RTX 2080 on 2026-10-31, with the release's build, one request at a time, with the files
the names fetch: Q8_0 for parakeet-tdt_ctc-0.6b-ja and reazonspeech-nemo-v2 on the 4,483 utterances of Common Voice 8.0's
Japanese test set (4,479 kept by VAD), one round, F16 for parakeet-tdt-0.6b-v3 on 300 English utterances of FLEURS (299
kept), the medians of three rounds, and Q8_0 for haas-ja-v1.0 on the 650 utterances of FLEURS' Japanese test split, the
means of two rounds. The RTFx is the seconds of audio over the seconds they took, so larger is faster.

| Model | Utterances | p50 latency, M5 / RTX 2080 | p90 latency, M5 / RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | Common Voice ja | 0.025 s / 0.047 s | 0.032 s / 0.071 s | 149 / 81 |
| reazonspeech-nemo-v2, beam search | Common Voice ja | 0.049 s / 0.096 s | 0.082 s / 0.183 s | 71 / 35 |
| reazonspeech-nemo-v2, greedy | Common Voice ja | 0.032 s / 0.048 s | 0.047 s / 0.087 s | 113 / 69 |
| parakeet-tdt-0.6b-v3 | FLEURS en | 0.050 s / 0.083 s | 0.076 s / 0.113 s | 152 / 99 |
| haas-ja-v1.0 | FLEURS ja | 0.042 s / 0.097 s | 0.063 s / 0.130 s | 205 / 98 |

A long recording is recognized whole. ReazonSpeech and parakeet-tdt_ctc-0.6b-ja use global attention, whose memory
grows with the square of the input length. Use `speech asr --vad silero-vad` to transcribe long recordings by region.
