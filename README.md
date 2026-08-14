# Transformixer

**Attention-based variate mixing without positional bias for multivariate time series forecasting**

Transformixer applies reversible instance normalization and a shared NLinear temporal forecast, then refines the result with a Transformer encoder over variate tokens **without** positional encodings, yielding a permutation-equivariant mixer with direct pairwise channel interactions. The design is inspired by prior work such as [xLSTM-Mixer](https://arxiv.org/abs/2410.16928) and [iTransformer](https://arxiv.org/abs/2310.06625).

This repository contains the project paper, slides, experiment notebooks, baseline implementations, and reported results.

## Highlights

- Drop-in Transformer variate mixer for the xLSTM-Mixer pipeline
- Permutation-equivariant cross-variate mixing (no positional encodings on channels)
- Competitive with strong baselines on Electricity (ECL) and Traffic
- Improves on xLSTM-Mixer on Traffic (best MAE among compared models); remains close on ECL

## Repository layout

```text
.
├── transformixer_paper/     # Paper (LaTeX + PDF), figures, analysis scripts
├── transformixer_slides/    # Beamer presentation
├── models/                  # Model codebases used in experiments
│   ├── transformixer/       # Proposed model (forked from xLSTM-Mixer)
│   ├── xlstm-mixer/
│   ├── itransformer/
│   ├── patchtst/
│   ├── informer/
│   └── rlinear/
├── notebooks/               # Training / evaluation notebooks (Colab-style)
├── papers/                  # Reference PDFs for baselines
└── experimental_results.json
```

## Method (short)

```text
Input → RevIN → Shared NLinear → Embed (H→d)
      → Transformer encoder (no PE, no causal mask)
      → Project (d→H) → RevIN denorm → Forecast
```

Temporal structure is handled by shared NLinear; cross-variate structure is handled by PE-free self-attention over variate tokens.

## Main results (\(L = H = 96\))

| Model         | ECL MSE | ECL MAE | Traffic MSE | Traffic MAE |
|---------------|---------|---------|-------------|-------------|
| RLinear       | 0.1973  | 0.2740  | 0.6463      | 0.3844      |
| Informer      | 0.3263  | 0.4068  | 1.0140      | 0.5624      |
| PatchTST      | 0.1754  | 0.2603  | 0.4699      | 0.3005      |
| iTransformer  | 0.1481  | 0.2398  | **0.3931**  | 0.2687      |
| xLSTM-Mixer   | **0.1480** | **0.2352** | 0.4157   | 0.2607      |
| Transformixer | 0.1517  | 0.2364  | 0.4068      | **0.2558**  |

Full metrics are also stored in [`experimental_results.json`](experimental_results.json). Figures and permutation/attention analyses live under [`transformixer_paper/figures/`](transformixer_paper/figures/).

## Paper and slides

| Artifact | Path |
|----------|------|
| Paper (PDF) | [`transformixer_paper/transformixer.pdf`](transformixer_paper/transformixer.pdf) |
| Slides (PDF) | [`transformixer_slides/transformixer_slides.pdf`](transformixer_slides/transformixer_slides.pdf) |

Analysis helpers:

```bash
cd transformixer_paper/scripts
python plot_results.py
python permutation_sensitivity.py
```

## Experiments

Before running any experiment notebooks, download the datasets as described in [Datasets](#datasets).

Training and evaluation were run via notebooks in [`notebooks/`](notebooks/), against the corresponding codebase under [`models/`](models/).

Each model directory retains its upstream setup instructions (`requirements.txt`, scripts, etc.). Transformixer and xLSTM-Mixer use a Lightning CLI; see [`models/transformixer/`](models/transformixer/) for installation details.

### Datasets

Experiments use the standard **Electricity (ECL)** and **Traffic** multivariate forecasting benchmarks (chronological 70%/10%/20% split).

Download `electricity.csv` and `traffic.csv` from [this Google Drive folder](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place them under `./datasets/`:

```text
./datasets/electricity.csv
./datasets/traffic.csv
```

### Scope

Results are from a constrained academic project setting: two datasets, one horizon (\(96\)), and a single run per model–dataset pair. Transformixer largely reuses xLSTM-Mixer training settings rather than a dedicated hyperparameter search.

## Acknowledgments

Baseline implementations and datasets are from their respective authors. This project builds directly on the [xLSTM-Mixer](https://arxiv.org/abs/2410.16928) codebase and experimental pipeline.
