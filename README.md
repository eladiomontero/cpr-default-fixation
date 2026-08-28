# CPR / CRD Default Fixation

*Paper in preparation.*

**What this measures:** whether LLMs anchor on a pre-filled "suggested" answer
instead of reasoning independently about a decision — the same *default
effect* documented in humans in
[Montero-Porras et al. 2025, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331348).
A model shown a plausible-looking pre-selected number and told "you can
change it if you want" should, if it's actually reasoning about the task,
land on an answer that depends on the game's incentives — not on whichever
number happened to be pre-filled. If its answer instead tracks the default
regardless of whether that default is fair, exploitative, or absurd, that's
default fixation, and it's a real exploitable failure mode: anyone who
controls the pre-filled field controls the model's decision.

This repo runs that test on 6 frontier/open models across **two different
game structures**, to check whether the effect is a property of one task or
a general property of how these models handle suggested answers:

| | [`cprd/`](cprd/) — CPR | [`CRD/`](CRD/) — CRD |
|---|---|---|
| Game | Common Pool Resource: extract 0–30 tokens from a shared pool | Collective Risk Dilemma: contribute {0,2,4} ECU to a shared threshold |
| Structure | Extract from a resource (more = worse for the group) | Contribute to a target (less = worse for the group) |
| Action space | 31 values, continuous-feeling | 3 values, sharply discrete |
| Benchmark anchors | 0 (abstain), 11 (social optimum), 18 (Nash), 23 (exploitative), 26 (indefensible), 30 (collapse) | 0 (free-ride), 2 (middle), 4 (full cooperation) |
| Human comparison | Yes — real participant data, incl. round-1 choices | Not collected |

Reversing the game direction (extract vs. contribute) is the point: if
default-following is a real property of how these models process suggested
answers, it should show up in both, not just the one they happened to be
tuned to expect.

## Headline result

At `default=26` in CPR — an anchor no benchmark defends — Qwen3 and gpt_5.1
place essentially all their probability mass exactly on 26, while Llama3.3
almost entirely ignores it:

![Spike heatmap: probability mass placed on the exact default value, by model and default](cprd/paper/figures/spike_heatmap.png)

The same wording/framing manipulation moves both games in the same
direction (see `framing_effect_bothgames.pdf` in both `cprd/paper/` and
`cprd/paper_wpull/`), and a distance-aware Wasserstein metric
(`cprd/paper_wpull/`) confirms the exact-match spike numbers aren't an
artifact of only counting mass that lands precisely on the anchor.

## Repo layout

```
.
├── cprd/                        # CPR game: pipeline, data, paper assets
│   ├── src/                     # Runners, logprob reconstruction, consolidation
│   ├── prompts/                 # System + task prompts (CPR, wording variants)
│   ├── data/models.json         # Model configs (provider, logprob support)
│   ├── output/                  # Per-run JSONs + consolidated CSVs
│   │   ├── mult_default/        # Main run: baseline + 6 defaults, 6 models
│   │   └── prompt_robustness/   # Wording-robustness sweep (V0_true..V4)
│   ├── paper/                   # Tables (.tex) + figures (.pdf/.png), net_pull metric
│   ├── paper_wpull/             # Same tables/figures, + distance-aware w_pull metric
│   └── README.md                # CPR-specific details, protocol, CLI usage
│
├── CRD/                         # CRD game: mirrors cprd/'s structure
│   ├── src/                     # Runners (reuse cprd/src's logprob/cache helpers)
│   ├── prompts/                 # System + task prompts (CRD, wording variants)
│   └── output/default_experiment/  # Per-run JSONs, comprehension test, summaries
│
└── .gitignore                   # Excludes: llm_cache.db, raw human-subjects data,
                                  # third-party PDFs, unrelated centipede/ project
```

## Key findings so far

- **Default fixation is real and large** in both games: models place
  anywhere from near-zero to near-total probability mass on a pre-filled
  value that has no independent justification.
- **Which models "fixate" is not fixed** — it depends on the exact wording
  of the suggestion sentence. A model that resists a default under one
  phrasing can flip to following it almost completely under a trivial
  reword (see `cprd/output/prompt_robustness/` and
  `cprd/paper/tables/wording_robustness.tex`).
- **The effect generalizes across game direction** (extract vs. contribute),
  though not identically — CRD's 3-option space makes the exact-match
  (`net_pull`) and distance-aware (`w_pull`) metrics track each other more
  loosely than in CPR's 31-option space.
- **Comprehension does not predict default-following** — models that
  understand the game rules perfectly (verified via a separate
  comprehension test in each game) fixate on defaults just as much as
  models that don't.
- **Humans move far less than models do**: human round-1 choices shift by a
  fraction of the distance to a suggested default; several models shift
  the *entire* distance or overshoot past it (see
  `cprd/paper/tables/human_vs_model_round1_pull.tex`).

## Reproducing / extending

Each game's pipeline is runnable from its own `src/` — see
[`cprd/README.md`](cprd/README.md) for the CPR CLI, output format, and
comprehension-check details; `CRD/src/` mirrors the same pattern (imports
`cprd/src`'s shared logprob/cache utilities rather than duplicating them).
All figures/tables are regenerated by the `build_*.py` scripts in
`cprd/paper/` and `cprd/paper_wpull/` from the already-collected CSVs — no
API calls needed to rebuild the paper assets from what's already here.

## Data note

`cprd/experiment/` normally also holds the raw human-participant CSV this
project's human benchmark is drawn from. That file is **excluded from this
repo** (`.gitignore`) because it contains MTurk/Prolific participant IDs —
only the aggregated, anonymized numbers derived from it are published here
(`derived_metrics*.csv`, the tables/figures, and the hardcoded benchmark
values cited in `cprd/README.md`).
