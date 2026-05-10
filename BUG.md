# Gemma4 nvfp4 Tool Call / Reasoning Token Leak

## Summary

When using `gemma4:26b-nvfp4` on macOS, ToolCall or Reasoning output can sometimes fail to split into Ollama's structured `tool_calls` / `thinking` fields and instead leak into assistant `content`.

One observed leaked suffix was:

```text
<|"|>,dump:true}<tool_call|>
```

In another case, reasoning content was not delivered as the separate `thinking` field. Instead, it appeared in the normal response body wrapped as a textual `thought` block. The behavior is intermittent: some requests with the same model and flow are parsed into tool calls / thinking correctly, while others expose raw special tokens or reasoning text in `content`.

## Environment

- Model: `gemma4:26b-nvfp4`
- Platform: macOS
- Engine path: safetensors / MLX runner path is likely involved for nvfp4 models
- API surface: `/api/chat`, tool calling, thinking/reasoning enabled or model-capability driven

## Expected Behavior

- Gemma4 tool call markup such as `<|tool_call>...<tool_call|>` must be consumed by Ollama's Gemma4 parser and emitted only as `message.tool_calls`.
- Gemma4 thinking markup such as `<|channel>thought\n...<channel|>` must be consumed by the parser and emitted only as `message.thinking`.
- Model-internal special tags should not appear in user-visible `message.content`.

## Actual Behavior

The response sometimes contains raw Gemma4 control syntax in `message.content`, for example:

```text
<|"|>,dump:true}<tool_call|>
```

This means the parser has already lost synchronization with the model's intended tool-call structure. Once that happens, Ollama does not emit a `tool_calls` entry for that request, so clients treat the response as normal assistant text instead of invoking a tool.

Reasoning can similarly leak when the model emits a non-canonical thinking block or a plain textual `thought` block rather than the exact Gemma4 channel format expected by the parser.

## Relevant Code Path

For `/api/chat`, Ollama initializes a built-in model parser from the model config:

- `server/routes.go`: `builtinParser = parsers.ParserForName(m.Config.Parser)`
- `server/routes.go`: `builtinParser.Add(r.Content, r.Done)` splits raw completion text into `content`, `thinking`, and `toolCalls`
- `model/parsers/parsers.go`: parser name `gemma4` maps to `Gemma4Parser{hasThinkingSupport: true}`
- `model/parsers/gemma4.go`: Gemma4-specific streaming parser

Gemma4 safetensors models are inferred as Gemma4 and receive Gemma4 parser / renderer settings during creation:

- `x/create/client/create.go`: `getParserName()` returns `gemma4` for Gemma4 architectures or model types
- `x/create/client/create.go`: `getRendererName()` returns `gemma4`
- `x/create/client/create.go`: config stores `Parser` and `Renderer`

For local GGUF creation, Gemma4 also gets parser / renderer defaults:

- `server/create.go`: architecture `gemma4` sets `config.Renderer = "gemma4"` and `config.Parser = "gemma4"`

## Suspected Root Cause

The Gemma4 tool-call parser treats the first occurrence of `<tool_call|>` in the tool-call buffer as the end of the tool call:

```go
if idx := strings.Index(bufStr, gemma4ToolCallCloseTag); idx != -1 {
    toolCallContent := bufStr[:idx]
    remaining := bufStr[idx+len(gemma4ToolCallCloseTag):]
    ...
}
```

This is fragile because it does not verify whether the close tag is structurally valid. In particular, it does not track:

- whether the parser is currently inside a Gemma string delimited by `<|"|>`
- whether braces / brackets are balanced
- whether a later close tag is the real tool-call boundary
- whether trailing fragments after a failed parse should still be suppressed as internal syntax

A malformed or quantization-affected model output can therefore desynchronize the parser. A plausible output shape is:

```text
<|tool_call>call:some_tool{arg:<|"|>...<tool_call|><|"|>,dump:true}<tool_call|>
```

The current parser would stop at the first `<tool_call|>`, try to parse a truncated tool call, and leave the remainder:

```text
<|"|>,dump:true}<tool_call|>
```

That remainder is not recognized as post-tool-call noise, because `Gemma4IgnoringPostToolCallNoise` only strips leading `<tool_call|>` and `<|tool_response>` markers. A fragment beginning with `<|"|>` is then passed through as assistant content.

## Why It Is Intermittent

The parser succeeds when the model emits the exact canonical format:

```text
<|tool_call>call:name{arg:<|"|>value<|"|>}<tool_call|>
```

The parser fails or leaks content when the model output is slightly malformed, truncated, duplicated, or reordered. This can happen nondeterministically due to sampling, context pressure, quantization error, or engine-specific decode behavior. nvfp4 is especially relevant because lower precision can increase format-token instability even when semantic output is mostly correct.

## Reasoning Leak Variant

Gemma4 thinking extraction expects:

```text
<|channel>thought
...
<channel|>
```

The parser strips the literal `thought\n` only immediately after `<|channel>`. If the model emits reasoning as plain Markdown/text such as:

````text
```thought
...
```
````

or emits a non-canonical channel sequence, the parser cannot identify it as internal thinking and will pass it through as normal content.

## MLX / nvfp4 Specific Concern

The MLX runner path appears less defensive than the llama runner path with respect to stop strings.

The regular llama runner checks configured stop sequences and truncates pending output before returning it. The MLX text generation pipeline primarily stops on tokenizer EOS and does not appear to apply the same arbitrary stop-sequence filtering in the shown path.

That means malformed Gemma4 tail tokens can continue far enough to reach the higher-level parser. The Gemma4 parser then becomes the last line of defense, and currently it is not robust enough for malformed tool-call fragments.

## Current Test Coverage

Existing Gemma4 parser tests cover many successful and repaired cases, including:

- normal tool calls
- streaming split tool calls
- missing terminal close tags on `done`
- extra `<tool_call|>` after a valid tool call
- `<|tool_response>` after a valid tool call
- some argument repair cases

However, the observed leak shape is not directly covered:

```text
<|"|>,dump:true}<tool_call|>
```

The missing coverage is specifically for a premature `<tool_call|>` inside or before a still-open Gemma string/object, followed by a raw Gemma string delimiter and additional argument-like text.

## Suggested Fix Direction

1. Make `Gemma4CollectingToolCall` structurally aware.
   - Do not accept the first `<tool_call|>` blindly.
   - Track Gemma string delimiter pairs (`<|"|>`).
   - Track object / array balance.
   - Treat `<tool_call|>` as final only when not inside a string and the call argument object is structurally complete.

2. Add targeted regression tests.
   - A test for premature close tag before `,<key>:...}<tool_call|>`.
   - A test for the exact leaked suffix shape beginning with `<|"|>`.
   - A streaming variant where the suffix is split across chunks.

3. Harden post-tool-call noise handling.
   - If a tool-call parse fails after seeing an open tag, do not immediately allow raw Gemma control fragments to become visible content.
   - Consider suppressing or quarantining known Gemma control fragments when they occur immediately after a failed tool-call parse.

4. Evaluate MLX stop-sequence handling.
   - Compare MLX generation behavior against `runner/llamarunner` stop handling.
   - If MLX does not apply configured stop strings, add equivalent stop-sequence buffering/truncation before chunks are returned to the server.

## Diagnostic Value

This bug is not just a client-side rendering issue. Once the special token fragment reaches `message.content`, the structured tool call has already failed at Ollama's parsing boundary. Clients cannot reliably recover because the intended tool name and arguments may have been discarded, truncated, or split from the leaked suffix.

The safest fix is therefore in Ollama's Gemma4 parser and, secondarily, in the MLX runner's stop-sequence handling.
