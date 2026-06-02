# MTP Task Sensitivity

MTP/speculative decoding is not a fixed multiplier. It speeds up decoding only
when the draft tokens are accepted by the target model. More speculative tokens
can improve synthetic decode throughput, but they can also waste work when the
task causes low acceptance.

The stable deployment rule for this fork is therefore:

- Use conservative MTP settings for mixed agent workloads.
- Treat higher MTP values as profile-specific speed options.
- Validate MTP with real prompts, not only repeated-token synthetic tests.

## Why Acceptance Matters

Speculative decoding drafts several future tokens, then the target model accepts
or rejects them. When many draft tokens are accepted, one target-model step can
produce multiple output tokens. When acceptance drops, the draft model still
consumes compute, but the target model accepts fewer tokens, so decode speed can
stagnate or regress.

This is why MTP performance depends on:

- task type and output entropy,
- model and draft quality,
- number of speculative tokens,
- KV dtype and graph/fallback path,
- whether the benchmark excludes cold-start/JIT overhead.

## LongGen3 4096/1024 Sweep

This sweep uses three Chinese long-generation prompts with about 4096 prompt
tokens and up to 1024 generated tokens each. Cold start, model load, graph
capture, and first-request JIT are excluded.

### Qwen3.6 27B

Qwen shows clear MTP scaling and then a plateau. MTP3 is the conservative
deployment point; MTP5 is only slightly faster on this specific sweep.

| MTP | Decode tok/s | Relative to noMTP |
|---|---:|---:|
| noMTP | 32.26 | 1.00x |
| MTP1 | 44.30 | 1.37x |
| MTP2 | 49.49 | 1.53x |
| MTP3 | 54.14 | 1.68x |
| MTP4 | 55.02 | 1.71x |
| MTP5 | 55.20 | 1.71x |

Per-case behavior at the top end is close: MTP5 reached `54.99 tok/s` on code,
`61.65 tok/s` on science, and `50.14 tok/s` on prose. MTP3 was slightly better
on prose in this run, which is why MTP3 remains the safer mixed-workload
default.

### Gemma4 31B

Gemma also benefits from MTP, but the gain is smaller and more workload
sensitive. MTP7 was numerically best in this sweep, while MTP8 was the first
decline.

| MTP | Decode tok/s | Relative to noMTP |
|---|---:|---:|
| noMTP | 30.67 | 1.00x |
| MTP1 | 21.06 | 0.69x |
| MTP2 | 27.06 | 0.88x |
| MTP3 | 31.19 | 1.02x |
| MTP4 | 33.81 | 1.10x |
| MTP5 | 35.81 | 1.17x |
| MTP6 | 36.71 | 1.20x |
| MTP7 | 37.68 | 1.23x |
| MTP8 | 36.81 | 1.20x |

The overall gain is real, but it is far from a universal 2x decode multiplier.
Gemma high-K MTP should be treated as a tuned profile, not a default rule.

## Three-Task 1024-Token Long-Output Probe

This probe used three shorter prompts capped near 1024 output tokens:

- a simple Python program,
- a Chinese romance story,
- a scientific-computing analysis.

The goal was to check whether higher MTP remains useful on realistic long
outputs rather than only synthetic repeated-token tests.

### Task 1: Simple Program

Code generation is the most MTP-friendly of the three tasks. The output has
strong local structure: imports, functions, loops, comments, and indentation are
easy for a draft path to predict. Higher K can therefore keep helping after the
mixed-task aggregate has already flattened.

| Model | noMTP | MTP1 | MTP2 | MTP3 | MTP4 | MTP5 | MTP6 | MTP8 | Best observed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 27B | - | - | 50.76 | 61.81 | 63.54 | 58.50 | - | 44.82 | MTP4, `63.54 tok/s` |
| Gemma4 31B | 8.10 | 49.20 | - | 79.51 | - | 97.56 | 109.33 | 73.12 | MTP6, `109.33 tok/s` |

Interpretation:

- Qwen improves sharply up to MTP3/MTP4, then higher K stops helping.
- Gemma benefits more on code than its aggregate result suggests; MTP6 was the
  fastest code row after warmup/JIT was excluded.
- MTP8 is not automatically better. It regressed for both models on this task
  despite being able to produce high synthetic numbers.

### Task 2: Romance Story / Natural Prose

Natural prose is the weakest MTP case. The model has many acceptable ways to
continue a sentence, choose adjectives, vary rhythm, or change phrasing. That
higher entropy reduces draft acceptance, so increasing K can waste draft work
instead of improving final decode speed.

| Model | noMTP | MTP1 | MTP2 | MTP3 | MTP4 | MTP5 | MTP6 | MTP8 | Best observed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 27B | - | - | 47.57 | 48.38 | 45.04 | 45.69 | - | 38.50 | MTP3, `48.38 tok/s` |
| Gemma4 31B | 33.07 | 42.33 | - | 49.56 | - | 48.62 | 46.52 | 42.64 | MTP3, `49.56 tok/s` |

Interpretation:

- This is the main reason mixed workloads should not blindly use very high K.
- Qwen's prose row peaks around MTP3 and declines after that.
- Gemma's prose row also peaks around MTP3; MTP5/MTP6 are better for code and
  science, but not for story-like output.

### Task 3: Scientific / Technical Analysis

Scientific and technical analysis sits between code and prose. It has natural
language, but also repeated structure: definitions, equations, steps, examples,
and summary bullets. In this probe it behaved much closer to code than to
romance prose.

| Model | noMTP | MTP1 | MTP2 | MTP3 | MTP4 | MTP5 | MTP6 | MTP8 | Best observed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 27B | - | - | 51.74 | 66.69 | 66.02 | 69.07 | - | 64.24 | MTP5, `69.07 tok/s` |
| Gemma4 31B | 33.77 | 53.31 | - | 83.41 | - | 103.85 | 104.79 | 102.37 | MTP6, `104.79 tok/s` |

Interpretation:

- Qwen keeps improving through MTP5 on this specific task, but MTP3 is already
  close enough to be the safer mixed default.
- Gemma's science row scales very well through MTP5/MTP6 and remains strong at
  MTP8, unlike the prose row.
- This task class is a good example of why MTP should be selected by workload,
  not by a single global K value.

### Aggregate View

The aggregate result is still useful, but only after looking at the task split.
It answers "what should the default mixed profile be?" rather than "which K is
best for every task?"

| Model | noMTP | MTP1 | MTP2 | MTP3 | MTP4 | MTP5 | MTP6 | MTP8 | Practical default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 27B | 36.92 | 44.56 | 49.96 | 57.87 | 56.51 | 56.12 | - | 46.99 | MTP3 |
| Gemma4 31B | 16.09 | 48.02 | - | 67.05 | - | 76.65 | 74.88 | 66.02 | MTP5 |

Qwen MTP8 produced much higher synthetic PP4096/TG128 numbers, but it regressed
on these realistic long-output prompts. The draft path did more work, yet
natural text did not accept enough draft tokens to justify the extra depth.

Gemma MTP5/MTP6 was very strong on code and scientific analysis. The same runs
were much weaker on natural prose: the romance/story row stayed around the
high-40 tok/s range because draft acceptance dropped. This is the clearest
example that MTP speed follows acceptance, not just the configured number of
draft tokens.

## Deployment Guidance

Use these settings as practical starting points:

| Route | Recommended MTP | Reason |
|---|---:|---|
| Qwen3.6 mixed agent workloads | MTP3 | Strong gain with less acceptance risk |
| Qwen3.6 code-heavy or synthetic speed checks | MTP4-MTP5 | Slightly higher peak, profile-specific |
| Qwen3.6 very high K such as MTP8 | Avoid by default | Can win synthetic tests but regress real long output |
| Gemma4 mixed workloads | MTP3-MTP5 | MTP5 is faster in tested long output; MTP3 is safer |
| Gemma4 code/science-heavy output | MTP5-MTP6 | Best observed task-class speed |
| Gemma4 natural prose-heavy output | Lower K preferred | Higher K can lose acceptance and flatten speed |

The benchmark result to trust depends on the serving target:

- Use PP4096/TG128 for clean single-request peak evidence.
- Use LongGen3 4096/1024 for balanced prefill+decode behavior.
- Use real agent/router tasks for service-like latency and quality.
- Always exclude cold start, model load, CUDA graph capture, Triton/AOT compile,
  and first-request JIT when comparing MTP settings.
