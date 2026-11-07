---
title: "Haas Japanese recognition on M2 Ultra with the MLX BF16 port"
date: 2026-11-07
lang: en
---

## Decision

The engineering decision for this product and workload is to keep the native F16 worker for this workload. MLX had the slower median latency on 9 of nine files. Each file has a ratio of its three-round median MLX latency to its three-round median native latency. 1.32066 is the median of those nine ratios. 1.216 to 1.480 is the range across those nine ratios. This is a product-level comparison using independently converted weights. Feature and decoder arithmetic may differ between the builds.

## Workload and builds

Inputs are nine distinct public FLEURS ja_jp test recordings. Durations span 6.36 to 28.20 s with three recordings in each of the 6–10, 10–20 and 20–30 s bands. All runs share one M2 Ultra host.

The MLX side runs the pinned port with the published BF16 weights on native arm64 Python 3.12.11 with mlx 0.32.4 and mlx-metal 0.32.4. Code is pinned at [haas-mlx cde96d2](https://github.com/matsuurajun/haas-mlx/tree/cde96d263f56aad75c1de70282f71a91562092dd) and weights at [Haas-ja-v1.0-mlx 15ef4f1](https://huggingface.co/mrjnzzz/Haas-ja-v1.0-mlx/tree/15ef4f11618c6a1ee50e0eafaae13beded8ecdf9).

The native side runs speech.cpp with the Metal encoder and the TDT decoder on CPU with four threads using F16 GGUF weights. Code is pinned at [speech.cpp 114c514](https://github.com/uchkw/speech.cpp/tree/114c514430c8aa1455c077819f8e72e674a48b75) and weights at [Haas-ja-v1.0-GGUF 6b965d5](https://huggingface.co/sakasegawa/Haas-ja-v1.0-GGUF/tree/6b965d5007ba780585de51549d57a8c1c0d67c7e).

## Method

Timed latency begins immediately before the first base64/JSON encoding and the first pipe send of the 32768-byte PCM transport. It ends after terminal text receipt. File reading and hash validation occur before it. No inference trace is active during timing. Warm figures are p50, p90 and p99 in ms plus RTFx where larger is faster. Each of the three rounds used a fresh process per side. Each side ran nine warmups followed by nine measured requests per round. That gives 27 measured requests per side and 54 total rows for the nine distinct recordings. Rounds ran in AB/BA/AB order with A as MLX. Fresh figures are the spawn-to-ready median, the first request median and the gross peak memory footprint in bytes. Spawn-to-ready covers process startup plus model and tokenizer preparation. It is not an isolated model-load time. The gross peak covers startup, loading, warmups and all measured requests. It is the maximum of three process peaks per product. No baseline is subtracted. It is not a GPU-only measure. Every fresh process received the longest 28.20 s input first. That first request was a warmup. It sits outside the 27 measured requests. The MLX first request in round 1 was 552.54 ms. The filesystem cache was not reset. The experimental paired harness is pinned at [speech.cpp 9fc6dcf](https://github.com/uchkw/speech.cpp/tree/9fc6dcf3f294aa5e447409fe31a1b79194d32510/experiments/haas-mac/mlx).

## Results

| Metric | MLX | Native |
| --- | --- | --- |
| Warm p50 (ms) | 108.43 | 73.28 |
| Warm p90 (ms) | 173.47 | 127.45 |
| Warm p99 (ms) | 184.24 | 144.55 |
| RTFx (larger is faster) | 136.99 | 183.11 |
| Fresh spawn-to-ready median (ms) | 258.44 | 259.54 |
| Fresh first request median (ms) | 213.59 | 159.49 |
| Gross peak memory footprint (bytes) | 1,992,787,936 | 1,348,552,408 |

## Output agreement

Raw plain text matched on 27 of 27 pairs. Both sides were individually stable across repeated runs. A separate public 7.98 s FLEURS smoke input outside the nine test recordings produced identical plain output from the resident process and the unmodified upstream MLX CLI.

These observations cover raw strings only. The full 650-recording test set, normalized CER for both products, raw token IDs and timestamps remain unmeasured. Keyword and segment comparison is out of scope.
