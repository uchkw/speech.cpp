# Trial 013: depthwise convolution, bias and SiLU stores

Status: source preparation only. No build, device check or timing has run. This is an experimental branch,
not an adopted speed-up. The tracking issue is https://github.com/uchkw/speech.cpp/issues/1.

## Source boundary

- Parent base: `114c514430c8aa1455c077819f8e72e674a48b75`.
- ggml runtime base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Cases first and original-kernel comparison: `97f3dcaf262818bfb52c4a0516fa76c2cb3c6b1c`.
- Experimental Metal implementation: `76f792b977196da0e4db64680a372e129ca06dba`.
- Parent branch: `codex/exp-haas-013`; ggml branch: `codex/haas-mac-013-dw-silu`.
- Original comparison branch: `codex/haas-mac-013-original-checks`, in
  `outputs/worktrees/haas-mac-013-original-ggml`.

The candidate and original comparison use identical test files. The ggml origin fetch URL is
`https://github.com/uchkw/ggml.git` and its push URL is `git@github.com:uchkw/ggml.git`. Only the coordinator pushes.
No earlier campaign runtime experiment is included. The parent runtime, GGUF layout, model weights, CPU and Vulkan
sources are unchanged. No persistent adoption decision or fork-main update is made here.

## Actual graph and arithmetic

`Encoder::convolution()` in `src/families/fastconformer/encoder.cpp` constructs a channel-fastest depthwise
convolution with a kernel of length 9, then a permutation, reshape, channel bias and SiLU. Haas has 24 such layers.
The converter already folds batch normalization into the depthwise weights and bias. The layout requires both
tensors to be F32, including in the Q4_K, Q8_0 and F16 model files. These are three affected model types, not
unmodified controls. Weight prepacking belongs to trial 014 and is not changed here.

The fusion matches exactly these five consecutive raw nodes:

```
CONV_2D_DW -> PERMUTE -> RESHAPE -> ADD -> UNARY(SILU)
```

It accepts one batch, one spatial row, more than one channel, kernel `[9,1,1,C]`, unit stride/dilation and width
padding 4. Input, packed kernel, bias and output stay F32. The convolution is the existing channel-fastest tiled
kernel: tap bounds, increasing tap order, accumulation expression, threadgroup selection and output strides are
unchanged. After accumulation, a safe-math, non-contracting float addition represents the separate F32 bias store.
The epilogue then uses the existing unary expression `value / (1 + exp(-value))` under the existing math mode.
This is the intended arithmetic boundary; compiled original/off/on equality is still unverified.

`GGML_METAL_DW_SILU=0|1` selects the experiment and defaults to 0. An invalid value aborts when the pattern is
examined. It is read once per process, so off/on comparisons use separate processes. The spatial convolution
kernel and the F16-weight convolution remain unfused. This is not a new M5-specific path; no Apple generation has
been measured for this trial.

The custom matcher applies `ggml_can_fuse_subgraph_ext()` to all five raw nodes, including both views. A protected
intermediate or any use outside the subgraph rejects fusion. Full validation additionally rejects any final-output
overlap with input, kernel or bias, even an exact alias. The existing general memory checks remain in force.
The new entry is excluded from allocator dependency insertion. Structural packing may still be followed by three
ordinary dispatches when the final allocation aliases an input. This preserves the existing allocator lifetime
policy; it can also reduce the number of real fusions and must be observed on the normal encoder graph.

## Checks prepared before implementation

`test-dw-silu-cases.h` is shared by the CPU-reference and direct Metal checks. It declares 50 cases:

- 23 eligible cases: C=1024 with T=1,2,3,4,5,8,9,17,31,32,33,171,375; C=31,32,33,255,256,257 with T=33;
  and offset, downstream consumer, signed weights and activations above the finite half range.
- 27 rejection controls: F16 weights at T=1,33,171; ReLU; protected convolution, permutation, reshape or sum;
  additional consumers of convolution views or sum; protected in-place activation; exact and partial input aliases;
  kernel and bias aliases; another reshape, full bias matrix, batch, stride, dilation, kernel size, missing reshape,
  strided bias/output/input and C=1.

`test-backend-ops -o CONV_DW_BIAS_SILU` contains 49 cases with the existing error bound. The padded channel-row
input is direct-Metal-only: the current CPU CWHN implementation indexes packed rows without using the input's
row strides. This exclusion is based on source inspection, not a newly executed CPU result. The strided bias and
output controls use row strides supported by the ordinary implementations.

`test-metal-dw-silu [expect-fusion: 0|1]` compares the same graph and buffers with generic fusion disabled and
enabled. Before each execution it restores all inputs and poisons output bytes not overlapping active inputs.
It verifies that poisoning preserves the inputs and that every output element differs from the previous result
before the next execution, including aliases restored from input data. It checks finite values, bitwise equality,
protected intermediate values, expected fusion counts and zero added allocation dependencies. Input/output FNV-1a
hashes permit original-kernel, candidate-off and candidate-on comparisons. Two artificial host-store guards show
that wrong-destination and skipped-last-store defects cannot pass through stale output. None has run yet.

Bias views in the fixtures are expanded before the convolution graph, keeping them outside the five-node pattern.
The real model uses an already loaded bias tensor. The direct check counts a fusion only when the compute encoder
accepts it, but its independent allocations do not establish that the normal model allocator accepts the same fusion.

## Deferred evaluation

All builds and GPU work wait for the coordinator's turn allocation and then hold
`/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock`, with builds limited to `-j4`.

1. Build original and candidate from clean, recorded source into separate artifact directories. Use the same test
   bytes, compare all direct input/output hashes in original/off/on processes, and require the declared fusion and
   poison checks. Preserve any failure; do not relax bitwise equality or reference bounds to obtain a pass.
2. Run the 49 CPU-reference cases and related existing depthwise/unary/allocator regressions. Record the separate
   direct-only padded-input control. Official F32 encoder reference and Windows/Vulkan validation remain separate
   requirements and are not replaced by synthetic CPU comparisons.
3. Run a normal Haas encoder graph that marks only the final projected tensor as output. Use graph-debug tracing
   separately from timing to count actual fused dispatches and compare projected output bytes. Do not initialize
   fusion statistics or set `GGML_METAL_FUSION_DEBUG` in this normal-command-buffer trace. Report the observed count;
   24 is a target, not an observed result, and alias rejection may reduce it.
4. If the checks pass, run the campaign's warm alternating 9-input, 3-round screen for Q4_K/Q8_0/F16, explicit CPU
   decoding with 4 threads, through `/usr/bin/python3` (3.9.6 on this machine). Record the binary SHA, explicit env
   and all inherited `GGML_*`/`SPEECH_*` settings using a frozen strict runner with the new allowlist entry. Compare
   request latency, stage time, output, memory and the fixed anchor. The runner checkpoint is not selected yet.
5. Present the screen before deciding whether full 30-input/650-quality and short6 measurements are justified.
   Any adoption requires a separate no-switch production patch and comparison with the original runtime. This
   source-only prototype establishes neither a gain nor numerical equivalence on Metal, Vulkan or another Mac.
