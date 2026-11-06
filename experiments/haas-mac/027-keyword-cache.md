# Retain one completed Haas keyword preparation per model

This source-only experiment starts at speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`, on `codex/exp-haas-027`. It contains no other runtime experiment and has
not been built, executed or measured. It does not adopt a product default or change the C API or worker protocol.

## What is reused

The model already owns its SentencePiece encoder, token strings and token lookup. A keyword request constructs a
`Vocabulary` with normalized words, readings, spellings and formatting forms. After plain decoding, and only where
`has_speech` says the audio has speech, it compiles each spelling's token forms into a `BoostingTree` before boosted
decoding. This trial retains those two prepared objects, not a tokenizer or a decoding result.

`SPEECH_HAAS_KEYWORD_CACHE` accepts `0` or `1` and defaults off. The engine reads it at construction only for the exact
Haas-ja-v1.0 model with KDA prediction and custom vocabulary support. Other models do not read it. The scope is host
preparation on either decoding backend; Metal, CPU and Windows/Vulkan execution still require their own checks.

The engine owns one `WordsCache`. Its key is the exact raw keyword string and the float alpha actually passed to
`Vocabulary`. The model's tokenizer and tree constants are fixed by that owner. Guard, audio, timestamps and decoding
device are not preparation dependencies. The tree itself does not depend on alpha, but alpha is part of the owned
Vocabulary, so an alpha change is a miss rather than a second cache tier. The key is never sorted, normalized or
deduplicated. Entry order, duplicate readings, NFC collisions and stable formatting precedence remain unchanged.

The existing line split, blank-only check, empty-keyword option checks and timestamps check run before cache lookup,
in their original order. A hit saves Vocabulary construction and tree compilation, not this inexpensive validation.
Every miss owns its pending preparation within the request. Tree construction stays after plain decoding and
`has_speech`. The engine publishes the pending entry only after the keyword path finishes without cancellation.
Parsing errors, timestamps errors, exceptions and cancellation do not replace a previous entry.

Only a preparation with a completed tree can be published. A first silent request keeps nothing, while a silent
request that hits an existing entry can reuse its Vocabulary without using its tree. A hit's tree is already complete
and is only read. This avoids changing a published entry during a cancelled request or temporarily retaining an
over-limit tree. A normal request with no keywords neither uses nor evicts the last keyword entry.

Plain decoding still runs with guard off. The audio test, optional boosted decoding, guarding and final formatting
are unchanged. TDT's current tree state, bonus/next vectors, KDA state and prediction outputs remain request-local.
No graph, arithmetic, weight, quantization or token decision is cached. The C API serializes requests of one model
with its existing model mutex; independent models own independent entries. No new mutex or global cache is added.

## Retention and lifetime

A successful pending entry is eligible for retention only when all of these hold:

- its raw keyword string is at most 4 KiB;
- it contains at most 128 nonblank entries before Vocabulary's normalization;
- its completed tree's allocated arc/state vector payload is at most 1 MiB, using vector capacities and `sizeof`.

Larger valid requests still use the ordinary request-local preparation and decoding. They are not rejected and do not
evict a retained small entry. The previous entry and one new pending entry can coexist during a miss. Publishing moves
ownership and releases the previous entry; model destruction releases the retained entry before the recognizer.

The tree payload limit is not a total heap limit. The raw key, Vocabulary strings/forms/container storage and allocator
overhead also consume memory. Report raw bytes, entry count and actual retained tree payload separately from process
peak memory. No custom allocator, per-spelling map or multi-entry cache is introduced.

## Check and measurement boundary

The dedicated checks are prepared separately from the runtime commit. Their construction/publication counters live
only in check executables; the runtime contains no counters, logs or hot timers. They have not been built or executed.

- `fastconformer-keyword-cache-check` needs no model or GPU. It exercises the real `WordsCache` ownership and admission
  with synthetic token forms: repeated A, A/B/A eviction, effective float alpha, raw ordering/duplicates/NFC, abandoned
  and invalid preparation, independent owners, exact and exceeded raw/entry bounds, exceeded allocated tree payload,
  and the disabled path. Synthetic trees do not certify SentencePiece or an official reference.
- `fastconformer-keyword-cache-path-check <model.gguf> <audio.wav> [cpu|gpu|device] [cpu|device]` needs Haas weights and
  speech audio. It uses the actual SentencePiece/tree path, observes construction/publication, and compares Vocabulary
  content, every tree field and every state's score/next vectors. It compares raw plain and boosted token IDs,
  emitted-token encoder frames and durations, graph counts and every WordsText field with uncached execution. It also
  runs the actual family engine off/on through repeated keywords, guard changes, invalid requests followed by a valid
  retry, empty keywords with timestamps, silence and cancellation of misses and retained hits. A cancelled hit in each
  decoding pass keeps the same entry, tree fields/payload and construction counts, then completes a repeated request.
  It checks early error category/input attribution
  and that invalid requests make no progress callback. No test inserts timers or counters into the engine.

Counters observe the production WordsCache/WordsModel path called directly by the check. The separate engine off/on
comparison checks observable results and errors; it does not introspect that engine's private cache counters.

For later serialized execution, the CPU F32 case is `fastconformer-keyword-cache-path-check "$F32_MODEL" "$AUDIO" cpu cpu`;
the quantized CPU decoder of a Metal model is the same command with `"$Q4_MODEL" "$AUDIO" gpu cpu`, and GPU decoding
uses `gpu device`. The path check fixes CPU threads at four. Its direct family-engine calls do not replace the existing
C API/worker recognition checks, which also check the request object's retry contract. It observes emitted-token
durations, not an instrumented trace of every blank/duration decision. Non-Haas selector controls remain unexecuted.

The existing vocabulary/word checks against official dumps remain separate. The words check accepts F32/F16 and
cannot certify Q4_K or Q8_0 official token identity. Quantized variants need their own same-weight differential check;
missing official inputs and Windows/Vulkan runs remain unverified. Original binary versus candidate-off entry-point
comparisons are also required because two candidate paths alone do not establish equivalence to the original binary.

Measure cold A, repeated A, alternating A/B and no-keyword requests separately. Count preparation and destruction in
end-to-end latency. Keep the CPU decoding count explicit at four, freeze the model/audio/runner and alternate off/on
under the coordinator's compute lease and shared lock for at least three rounds. Separate preparation time from whole
request RTFx, load/first-request cost and peak memory. Any preparation timing belongs in an experiment tool only.

The normal 650-input timestamp quality run has no keywords and is a regression control. Keyword quality is a separate
condition with timestamps off and a dedicated raw-decoding check. A keyword cache gain is not a gain of the primary
no-keyword campaign and is not added to its cumulative code speed-up. This checkpoint has no correctness, performance
or adoption verdict.
