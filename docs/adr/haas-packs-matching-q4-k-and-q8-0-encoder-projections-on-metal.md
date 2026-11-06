# Haas packs matching Q4_K and Q8_0 encoder projections on Metal

## Context

Combining the encoder's Q, K and V projections reduces three matrix products to one but retains a second copy of their
weights. A prototype on an M2 Ultra with four CPU decoder threads on 2026-11-07 reduced Haas-ja-v1.0's paired median
request latency by 2.48% in Q4_K and 1.62% in Q8_0 over 30 inputs and three rounds, with identical public outputs on
650 quality inputs. F16 did not show a stable gain. Other Metal devices and CPU and Vulkan encoders have no such
performance evidence. The family permits different storage types for individual matrices, including Q, K and V.

## Decision

Automatically pack the Metal encoder of a KDA model with unbiased global attention only when every layer's Q, K and V
have the same Q4_K or Q8_0 type. Keep the existing separate projections for every other model, type or backend. This
selects an optimization; it adds no requirement to the GGUF contract. The packed copy keeps the original weight bytes
and belongs to the encoder, so no experiment switch, new model file or request option is needed.

The alternatives were turned down:

- Applying packing to every type and backend. The memory cost is certain, while F16 has no stable measured gain and
  the other cases have no measured benefit.
- Requiring every valid file to have matching Q, K and V types. The layout already permits mixed types independently.
- Converting mixed matrices to one type. It changes the precision the file selects and the work being evaluated.

## Consequences

The standard Haas-ja-v1.0 files retain another 40.5 MiB in Q4_K and 76.5 MiB in Q8_0 while the model is loaded, with an
additional copy at load time. Performance on Metal devices other than the M2 Ultra remains unmeasured.
