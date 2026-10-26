# Can Pricing Agents Collude Without Words?

Code and recorded data for experiments with Clef-Flash and Qwen3.5-9B in a repeated two-firm pricing game.

## Run

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The command checks the saved data, recomputes the results, and generates six figures in `outputs/`. Runs on CPU; no model downloads or API calls.

## Experiments

Both models were tested against live rivals, with rival-price fields hidden, and against recorded rival prices.

| Model | Runs per setting | Rounds | Recorded rival paths |
| --- | ---: | ---: | ---: |
| Clef-Flash | 20 | 80 | 4 |
| Qwen3.5-9B | 12 | 60 | 3 |

Run sizes were chosen from measured speed under a fixed compute budget. The data also include paired price-cut interventions and fixed-input checks. Shared rival paths and paired continuations are accounted for in the analysis.

## Files

- `src/` — market, model adapters, analysis, and plotting code.
- `evidence.zip` — completed experiment data and executed source snapshots.
- `provenance.json` — model revisions, settings, and checksums.
- `reproduce.py` — verification and reanalysis entry point.
