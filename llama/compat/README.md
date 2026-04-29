# llama.cpp compatibility shim

This directory holds an in-process compatibility layer that lets upstream
`llama-server` load GGUFs produced by older versions of Ollama (and files
pulled from the Ollama registry) without re-converting or re-downloading.

The layer is applied automatically at build time via CMake `FetchContent`'s
`PATCH_COMMAND` for normal fetched builds — there is no separate "apply
patches" step. If CMake is pointed at a source override through
`FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`, the same patch is applied directly during
configure. If `OLLAMA_LLAMA_CPP_SOURCE` is set, the patch is intentionally not
applied so a developer can iterate on a local llama.cpp tree explicitly.

## Files

- `llama-ollama-compat.h`, `llama-ollama-compat.cpp` — the shim itself. These
  are regular source files owned by Ollama; they are linked into the fetched
  `llama` and `mtmd` targets from this directory with `target_sources`.
- `llama-ollama-compat-util.h`, `llama-ollama-compat-util.cpp` — shared helper
  code for KV edits, tensor renames, skip-prefix tracking, tensor load ops,
  and small tensor repacking primitives.
- `upstream-edits.patch` — small additive edits to upstream files so the
  shim gets called. It currently touches only `src/llama-model-loader.cpp`
  and `tools/mtmd/clip.cpp`. Kept as a real `git` patch so re-generation on
  upstream bumps is one command.
- `compat.cmake`, `apply-patch.cmake` — CMake glue and an idempotent patch
  applier used by `llama/server/CMakeLists.txt`.

This directory replaces the old `llama/patches` stack for llama-server builds;
the active upstream edit surface is `upstream-edits.patch` plus the
Ollama-owned source files above.

## What the shim does

The shim runs at a small set of loader hook points:

1. **Main model constructor, after the architecture is read**:
   `translate_metadata` inspects the just-parsed metadata and decides whether
   the file is an Ollama-format GGUF. If so, it mutates the in-memory
   `gguf_context` and `ggml_context` (KV names, tensor names, tensor types,
   tensor shapes, and selected tokenizer metadata) so the rest of the loader
   sees an upstream-shape file. It can also request mmap disablement when a
   handler needs writable backend buffers for transformed tensor data.

2. **Main model tensor indexing**: `should_skip_tensor` hides embedded
   projector, vision, audio, MTP, or other Ollama-only tensors from the text
   loader's weights map.

3. **Main model tensor load loop**: `maybe_load_text_tensor` applies registered
   text-side load operations, such as FFN concat or dtype promotion, before
   the normal upstream file read.

4. **`mtmd/clip` constructor**: `translate_clip_metadata` rewrites a
   clip-facing view of monolithic Ollama GGUFs into upstream mmproj shape.
   It also handles legacy LLaVA/BakLLaVA projectors that need a default
   `clip.projector_type`.

5. **`mtmd/clip` tensor load loop**: `maybe_load_tensor` applies clip-side
   load operations, such as F16 to F32 promotion, QKV merge, tensor repack,
   tensor split, or zero-fill.

Non-Ollama files are detected per architecture by the absence of the legacy
Ollama markers each handler expects. When no handler matches, every compat
entry point is a no-op.

## Currently supported architectures

This table tracks the dispatch surface. The per-handler comments in
`llama-ollama-compat.cpp` are the source of truth for exact KV/tensor maps.

| Internal arch / marker | Text loader handling | Clip/mmproj handling |
|---|---|---|
| `gemma3` | Legacy Gemma 3 metadata, tokenizer, and embedded vision/projector cleanup. | `gemma3` projector translation. |
| `gemma3` + embedding markers (`embeddinggemma`) | Rewrites to upstream `gemma-embedding`, fixes embedding-specific KVs and dense/norm tensors. | n/a |
| `bert` + Snowflake markers (`snowflake-arctic-embed2`) | Fixes legacy Snowflake Arctic Embed 2 tokenizer metadata. | n/a |
| `gemma3n` | Normalizes tokenizer/EOS metadata, truncates vocab-shaped tensors, and drops unused embedded vision/audio/projector tensors. | n/a |
| `gemma4` | Drops audio/vision/projector tensors from the text view while audio remains unsupported upstream. | `gemma4` projector translation. |
| `gptoss` | Renames to upstream `gpt-oss`, copies KVs, injects missing expert FFN metadata, and renames tensors. | n/a |
| `lfm2` | Renames stale norm tensors and fixes feed-forward metadata. | n/a |
| `olmo3` | Rewrites to the upstream OLMo2-compatible loader path. | n/a |
| `mistral3` | Fixes RoPE/YaRN metadata and hides embedded vision/projector tensors. | Pixtral-style projector translation. |
| `qwen35`, `qwen35moe` | Fixes Qwen3.5/Qwen3-VL-style text metadata and hides embedded vision/projector/MTP tensors. | Qwen3-VL merger-style projector translation. |
| `qwen25vl` | Rewrites to upstream `qwen2vl` metadata conventions. | Qwen2.5-VL projector translation. |
| `qwen3vl`, `qwen3vlmoe` | Adds missing Qwen3-VL metadata and hides embedded vision/projector tensors. | Qwen3-VL projector translation, including QKV merge and patch-embedding split/repack. |
| `deepseekocr` | Rewrites to upstream `deepseek2-ocr`, injects missing OCR/MoE metadata, and hides embedded SAM/vision/projector tensors. | DeepSeek OCR projector translation. |
| `glmocr` | Rewrites GLM OCR metadata/tensors to the upstream-compatible view. | GLM OCR projector translation. |
| `glm4moelite` | Rewrites GLM-4.7 Flash MLA metadata to the upstream `deepseek2` path and fixes special-token metadata. | n/a |
| `nemotron_h_moe` | Fixes latent-FFN variants and hides MTP tensors. | n/a |
| `nemotron_h_omni` | Rewrites to the selected Nemotron text loader and hides audio/vision/projector tensors; audio remains intentionally hidden. | Nemotron V2 VL projector translation with audio disabled. |
| `llama` legacy Llama 3 markers | Fixes legacy tokenizer metadata. | n/a |
| `llama4` | Hides embedded vision/projector tensors from the text view. | Llama 4 projector translation. |
| legacy `clip` projector without `clip.projector_type` | n/a | Defaults legacy LLaVA/BakLLaVA projectors to `clip.projector_type=mlp`. |

Usage:

```
llama-server --model /path/to/ollama-blob --mmproj /path/to/ollama-blob
```

Passing the same monolithic GGUF as both `--model` and `--mmproj` works —
each loader applies its own translation.

Additional architectures are added by implementing a `handle_<arch>()`
and (for vision models) `handle_<arch>_clip()` in `llama-ollama-compat.cpp`
and dispatching them from `translate_metadata` / `translate_clip_metadata`.
For monolithic vision models, also update the `compatClipArches` allowlist in
`llm/llama_server.go` so Ollama passes the main GGUF as `--mmproj`.

## Regenerating `upstream-edits.patch`

After upstream changes the insertion points (rare), re-apply the edits to
a fresh checkout and run:

```
cd /path/to/llama.cpp
git diff -- \
    src/llama-model-loader.cpp \
    tools/mtmd/clip.cpp \
    > /path/to/ollama/llama/compat/upstream-edits.patch
```

## Why not fork llama.cpp or vendor it?

Forking means tracking upstream manually. Vendoring means snapshotting all of
llama.cpp's source in the Ollama tree (the old `llama/llama.cpp/` layout).
This shim keeps upstream unmodified on disk and the Ollama-specific logic
isolated in a few Ollama-owned source files plus a small diff — upstream bumps
are usually just `LLAMA_CPP_VERSION` changes and, when insertion points move,
`upstream-edits.patch` regeneration.

## Maintenance: non-public API dependencies

The compat code is mostly written against stable public APIs (`gguf.h`,
`ggml.h`, `ggml-backend.h`). There are three places where we lean on
something that isn't strictly public:

| Hack | Why | Escape hatch if upstream changes |
|---|---|---|
| Direct writes to `ggml_tensor::type` / `ne[]` / `nb[]` | No sanctioned mutator exists for post-creation tensor reshape/retype. Struct is public so this works today. | Ask upstream to expose `ggml_tensor_set_{type,shape}` helpers, or introduce them in our compat util and submit a PR. |
| `const_cast<char *>(gguf_get_tensor_name(...))` in `rename_tensor` | Pointer aims into a mutable `char[GGML_MAX_NAME]` buffer inside a `std::vector` element; the const is API hygiene. Lets us rename gguf tensors without a new public helper. | Add `gguf_rename_tensor` to `gguf.h` (10 lines) and drop the `const_cast`. |
| `llama_model_loader` forward-decl from `src/llama-model-loader.h` | Used only as an opaque pointer key for per-loader skip-prefix, mmap-disable, and source-path registries. Never dereferenced. | Replace with `const void *` in our registry signatures. Zero behavioral change. |

None of these have changed in years. If an upstream bump breaks any of
them, each has a trivial workaround. See the top of
`llama-ollama-compat-util.h` for the inline notes.

## Documented hacks inside per-arch handlers

- **`reclaim_slot_as` (patch-embedding splits)** — repurposes an orphaned
  tensor slot as a newly synthesized tensor when a clip handler must split a
  source tensor into multiple upstream-facing tensors. Needed because
  clip.cpp's `ctx_meta` is sized for exactly the original tensor count
  (no_alloc branch of `gguf_init_from_file` uses
  `n_tensors * ggml_tensor_overhead()` with zero slack). Comment in the helper
  and call sites explains the reasoning; replacement would be a small upstream
  patch that adds slack to the ctx size.

- **Load-op registry overrides `file_offset`** — `maybe_load_tensor` gets
  passed the gguf offset by its caller but ignores it when a registered
  op exists. Intentional: the ops capture their own source offsets at
  translate time (before our renames invalidate them). Documented in the
  op-registration helpers.
