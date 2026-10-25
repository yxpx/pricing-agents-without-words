# Can Pricing Agents Collude Without Words?

Reproducibility files for Yuvraj Pandey's study at the [AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25), October 2026.

**Run the analysis on an ordinary CPU. No API key, model download, paid service, or GPU is needed.** The download contains the completed experiments' synthetic market records, analysis code, and figure code. It does not contain the paper, private workspace files, remote-access tools, model weights, or credentials.

## Reproduce the results

Use Python 3.12, then run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Allow roughly a minute or two for analysis, depending on the CPU; dependency installation is separate. The small compressed archive expands to about 41 MB. The outputs use additional disk space.

The command checks archive and source hashes, unpacks the saved records, tests the market calculations, rechecks every recorded payoff and observation, recomputes the main estimates, and rebuilds all five charts. It also checks that the main estimates agree with the original saved results. Outputs appear in `outputs/`:

- `reproduction_report.json`: completion status, sample counts, and primary results.
- `clef_summary.json` and `qwen_summary.json`: numerical results and uncertainty estimates.
- `clef_validation.json` and `qwen_validation.json`: data-integrity checks.
- `comparison.json`: the secondary comparison using equal-length runs.
- `figures/`: five vector PDF charts and their numerical source values.
- `previews/`: PNG copies of the charts.

This is **reanalysis of saved experiments**, not a new model experiment. It cannot reproduce the randomness of a new GPU run without running the models again. Fonts may fall back from Times New Roman to STIX on machines without Times; numerical results do not depend on that choice.

## What was tested?

Two simulated firms repeatedly chose one of eleven prices, from 1.250 to 2.000. Clef-Flash used its native decision head; Qwen3.5-9B was restricted to scoring price-option identifiers. Neither generated messages, written reasoning, or plans, and neither updated its weights.

Each model had the **same three settings**:

1. **Live:** both firms respond to recent prices and their own sales/profits.
2. **Own history:** both respond, but the direct rival-price field is hidden. Sales still contain information about the rival, so this does not remove all information.
3. **Recorded rival:** one firm faces a fixed price sequence from a separately generated market. The code calls this `replay` and its source sequences `donors`.

| Completed study | Market repeats per setting | Rounds per repeat | Separate rival sequences | Additional continuations |
| --- | ---: | ---: | ---: | ---: |
| Clef-Flash | 20 | 80 | 4 | 40 |
| Qwen3.5-9B | 12 | 60 | 3 | 24 |

The unequal sizes came from a shared runtime budget and a short speed/numerical-stability check. Qwen required higher numerical precision and ran more slowly, so its smaller design was selected before inspecting final results. The settings themselves were identical. Primary comparisons stay within each model; a separate, secondary analysis aligns both models to 60 rounds. Different training, interfaces, precision, and original horizons prevent attributing differences to architecture alone.

## How much data?

The archive contains **96 main market repeats**, seven separate source markets for recorded rivals, and **64 matched continuations**. It contains 17,240 firm-round rows and 14,308 actual model evaluations, plus 68 fixed-state diagnostic records. Each market-round has two firm rows; these rows are not independent experimental repeats. Some rows describe scripted rival choices rather than model calls.

The final-quarter analysis uses the last 20 rounds for Clef and the last 15 for Qwen. Each live market was restarted from its saved history in two additional versions: one with a one-round rival undercut, and one with a rival held at the competitive price. These versions share a parent market and must remain paired in the analysis.

## What does the result mean?

The competitive benchmark is price 1.475; the price maximizing the firms' combined profit is 1.925. Average live outcomes stayed near the competitive benchmark. The main live-minus-recorded-rival profit intervals included zero for both models. Supplied high-price histories nevertheless moved both models' next choices toward high prices.

This does not establish either collusion or safety. There are only three or four source sequences for recorded rivals, the model runs are short, and fixed-history diagnostics do not measure whether a convention would persist. There are **no ground-truth collusion labels**. Similar or high prices alone do not establish a cartel.

## Files and provenance

```text
README.md             This guide and the experiment's interpretation limits
requirements.txt      CPU-only dependencies tested for this release
reproduce.py          One-command verification, analysis, and plotting
provenance.json       Archive/member/source hashes, revisions, settings, counts
src/                  Market, runner, inference adapter, diagnostics, analysis, plots
evidence.zip         Completed synthetic data and exact executed source snapshots
```

`evidence.zip` contains only completed primary Clef and Qwen runs, their diagnostics, and four exact per-model source snapshots. Pilots and the superseded incomplete A100 run are excluded. Full prompts, allowed prices, probabilities, actual choices, sales, profits, condition assignments, seeds, model revisions, and recorded-rival identities are retained. The model weights are not redistributed.

The release's plotting code differs from the workspace copy only where needed to read `data/` and write `outputs/`. The analysis and inference modules retain the study implementation. `src/adapters.py`, `runner.py`, and `diagnostics.py` document how the original GPU experiments were implemented, but `reproduce.py` never invokes GPU inference. Re-running models is a separate, expensive operation requiring the original GPU dependencies and model downloads; it is not needed to check this release's results. The generated data came from [Cloudflare/Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) and [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), not TypeSafe Jev.

Checksums establish file identity, not the validity of a scientific interpretation. AI tools assisted with implementation and checks; author review remains necessary.
