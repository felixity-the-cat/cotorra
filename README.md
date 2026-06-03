[![DOI](img/1193885071.svg)](https://doi.org/10.5281/zenodo.20414127)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/bbj-lab/cotorra/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/bbj-lab/cotorra)

# Cotorra: a configurable trainer

> 🦜 the wild parakeet of Chicago's south side

<img src="img/cotorra.png" width="400" style="display: block;
margin: 0 auto; -webkit-mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );
  mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );"/>

## About

This repo provides a configurable trainer for generative event models on
tokenized timelines. _Cotorra_ is a Spanish term for a small-to-medium sized
parrot, particularly the Monk parakeet. Monk parakeets were introduced to the
south side of Chicago, where they have flourished. [^1] It benefits from previous
experience training foundation models on tokenized electronic health records.
[^2] [^3] [^4] [^5]

## Installation

You can download and install this package as follows:

```sh
git clone --branch config-refactor git@github.com:bbj-lab/cotorra.git
cd cotorra
python -m venv .venv
. .venv/bin/activate
pip install -e ".[gen]" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

## Context

Suppose you have a dataset of tokenized timelines `tokens_times.parquet` as a
parquet table with columns:

- `subject_id`
- `tokens` — the integer token sequence for the subject's timeline.
- `times` — a parallel list of timestamps, one per token, indicating when each
  event occurred.

The table will look something like this:

```
┌────────────────────┬─────────────────┬─────────────────────────────────┐
│ subject_id         ┆ tokens          ┆ times                           │
│ ---                ┆ ---             ┆ ---                             │
│ str                ┆ list[u32]       ┆ list[datetime[μs]]              │
╞════════════════════╪═════════════════╪═════════════════════════════════╡
│ 20002103           ┆ [20, 350, … 21] ┆ [2116-05-08 02:45:00, 2116-05-… │
│ 20008372           ┆ [20, 350, … 21] ┆ [2110-10-30 13:03:00, 2110-10-… │
│ …                  ┆ …               ┆ …                               │
│ 29994865           ┆ [20, 364, … 21] ┆ [2111-01-28 21:49:00, 2111-01-… │
└────────────────────┴─────────────────┴─────────────────────────────────┘
```

You also have a `tokenizer.yaml`, a plain yaml file that contains information
about the configuration, learned vocabulary, and bins. This file is sufficient to
reconstitute the tokenizer object. We only need this file to contain a lookup
table:

```yaml
lookup:
  UNK: 0
  ADMN//direct: 1
  ADMN//ed: 2
  ADMN//elective: 3
  AGE//age_Q0: 4
  ...
```

Finally, we need `subject_splits.parquet` which is a table listing out all
subject_id's and their corresponding split assignment (with splits: `train`,
`tuning`, and `held_out`):

```
┌────────────┬──────────┐
│ subject_id ┆ split    │
│ ---        ┆ ---      │
│ str        ┆ str      │
╞════════════╪══════════╡
│ 21081215   ┆ train    │
│ 20302177   ┆ train    │
│ …          ┆ …        │
│ 28150003   ┆ held_out │
│ 22151813   ┆ held_out │
└────────────┴──────────┘
```

<!-- prettier-ignore-start -->
> [!TIP]
> For getting your data to this point, check out our configurable
> collator / tokenizer: [☕️ cocoa](https://github.com/bbj-lab/cocoa)
<!-- prettier-ignore-end -->

Given these things, we want to train a model to predict the next token in a
subject's timeline given their complete history or context up to this point. This
package is designed to do that in a configurable way.

## Configuration

This library can be extensively customized through yaml configuration files. Each
command has its own default config under `src/cotorra/config/`, which you can
override by passing a config file via the appropriate CLI flag. Any value can
also be overridden programmatically via `**kwargs` which are merged on top of the
YAML config via OmegaConf.

#### Training configuration ([example](src/cotorra/config/training.yaml))

Used by `cotorra train` and `cotorra tune`.

- **model_name**: Name or path of the HuggingFace model (e.g.,
  `meta-llama/Llama-3.2-1B`).
- **model_args**: Model architecture parameters passed directly to HuggingFace's
  [`AutoConfig`](https://huggingface.co/docs/transformers/en/model_doc/auto).
- **max_seq_len**: Maximum sequence length for model input.
- **n_epochs**: Number of epochs (handled in the dataloader, not the trainer).
- **run_name**: Name for the current run (referenced by `wandb` and
  `training_args`).
- **tokens_of_interest**: List of special tokens to upweight during training
  (referenced by loss config).
- **wandb**:
  - **project**: Weights & Biases project name for experiment tracking.
  - **run_name**: Name for the current run.
- **custom_loss**: Boolean flag to enable custom loss functions (default:
  `false`).
- **quantile_token_loss** _(optional)_: Upweights loss on quantile boundary
  tokens.
  - **qt_weight**: Weight multiplier for quantile tokens.
- **label_weighted_loss** _(optional)_: Upweights loss on specific tokens of
  clinical interest.
  - **tokens_of_interest**: List of token labels to upweight.
  - **toi_weight**: Weight multiplier applied to those tokens.
- **time_based_rope** _(optional)_: Enables time-aware rotary position
  embeddings.
  - **sec_per_pos_id**: Number of seconds represented by one position id
    increment.
- **training_args**: Arguments passed to HuggingFace's
  [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).
- **tuning_args**: Arguments passed to HuggingFace's
  [`hyperparameter_search`](https://huggingface.co/docs/transformers/hpo_train?backends=Optuna)
  when `cotorra tune` is called.

#### Extraction configuration ([example](src/cotorra/config/extraction.yaml))

Used by `cotorra extract`.

- **max_seq_len**: Maximum sequence length.
- **time_based_rope** _(optional)_: Enables time-aware position ids during
  extraction (must match the setting used at training time).
  - **sec_per_pos_id**: Number of seconds represented by one position id
    increment.
- **extract**:
  - **max_len**: Maximum input length (tokens) during extraction.
  - **batch_size**: Batch size for inference.
  - **shard_size** _(optional)_: Number of samples per output parquet shard. Omit
    to write a single file per split.

#### Scoring configuration ([example](src/cotorra/config/scoring.yaml))

Used by `cotorra generative-score` and `cotorra rep-based-score`.

- **run_name**: Name for the current run, used to label output files.
- **tokens_of_interest**: List of token-based outcomes of interest.
- **score**:
  - **max_len**: Maximum input length (tokens) during scoring.
  - **n_samp**: Number of Monte Carlo samples per input per trajectory type.
  - **target_tokens**: Token-based outcomes of interest to score.
  - **end_tokens**: Tokens that naturally terminate a generated sequence (e.g.
    `EOS`).
  - **suppressed_tokens**: Tokens to suppress via logit bias during generation
    (e.g. `PAD`).
  - **trunc_id**: Token id forced after the time horizon is exceeded.
  - **max_time**: Maximum time horizon in minutes.
  - **batch_size**: Batch size for inference.

## Usage

To use a different dataset or schema, create new YAML config files and pass them
via the appropriate CLI flag, or pass your options directly to the class. All
classes accept `**kwargs` that are merged on top of the YAML config via
OmegaConf, so any config value can be overridden programmatically:

```python
from cotorra.trainer import Trainer

trainer = Trainer(
    processed_data_home="~/other/data",
    output_home="~/other/output",
)
trainer.train()
```

### CLI

We provide a CLI:

```
 Usage: cotorra [OPTIONS] COMMAND [ARGS]...

 Configurable training for generative event models (vXX.X.X)

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.     │
│ --show-completion             Show completion for the current shell, to     │
│                               copy it or customize the installation.        │
│ --help                        Show this message and exit.                   │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ train             Train a model on tokenized data. For tokenization,        │
│                   consult the cocoa package.                                │
│ tune              Run hyperparameter tuning while training a model.         │
│ extract           Extract representations from a trained model.             │
│ generative-score  Generate SCORE/REACH metrics from a trained model and     │
│                   save them to parquet.                                     │
│ rep-based-score   Generate rep-based scores for the token-based outcomes of │
│                   interest.                                                 │
│                   Note: this requires that features have already been       │
│                   extracted and saved                                       │
╰─────────────────────────────────────────────────────────────────────────────╯
```

with commands:

- `cotorra train`

  ```
  Usage: cotorra train [OPTIONS]

  Train a model on tokenized data. For tokenization, consult the cocoa package.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │    --training-config      -t      PATH  Training configuration file         │
  │                                         (overrides default)                 │
  │ *  --processed-data-home  -p      TEXT  Processed data directory (overrides │
  │                                         config)                             │
  │                                         [required]                          │
  │ *  --output-home          -o      TEXT  Output directory for trained models │
  │                                         [required]                          │
  │    --verbose              -v            Verbose logging for collate         │
  │    --help                               Show this message and exit.         │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra tune`

  ```
  Usage: cotorra tune [OPTIONS]

  Run hyperparameter tuning while training a model.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │    --training-config      -t      PATH  Training configuration file         │
  │                                         (overrides default)                 │
  │ *  --processed-data-home  -p      TEXT  Processed data directory (overrides │
  │                                         config)                             │
  │                                         [required]                          │
  │ *  --output-home          -o      TEXT  Output directory for trained models │
  │                                         [required]                          │
  │    --verbose              -v            Verbose logging for collate         │
  │    --help                               Show this message and exit.         │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra generative-score`

  ```
  Usage: cotorra generative-score [OPTIONS]

  Generate SCORE/REACH metrics from a trained model and save them to parquet.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │    --scoring-config       -s      PATH  Scoring configuration file          │
  │                                         (overrides default)                 │
  │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
  │ *  --model-home           -m      TEXT  Directory of the trained model to   │
  │                                         score with                          │
  │                                         [required]                          │
  │    --output-home          -o      TEXT  Output directory for scores,        │
  │                                         defaults to processed-data-home     │
  │    --verbose              -v            Verbose logging for collate         │
  │    --help                               Show this message and exit.         │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra extract`

  ```
  Usage: cotorra extract [OPTIONS]

  Extract representations from a trained model.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │    --extraction-config    -e      PATH  Extraction configuration file       │
  │                                         (overrides default)                 │
  │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
  │ *  --model-home           -m      TEXT  Directory of the trained model to   │
  │                                         extract from                        │
  │                                         [required]                          │
  │    --output-home          -o      TEXT  Output directory for extracted      │
  │                                         features, defaults to               │
  │                                         processed-data-home                 │
  │    --all-times            -a            Extract features for all time steps │
  │                                         (instead of just the final one)?    │
  │    --help                               Show this message and exit.         │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra rep-based-score` (note: you need to run `extract` first)

  ```
  Usage: cotorra rep-based-score [OPTIONS]

  Generate rep-based scores for the token-based outcomes of interest. Note:
  this requires that features have already been extracted and saved

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │    --scoring-config       -s      PATH  Scoring configuration file          │
  │                                         (overrides default)                 │
  │ *  --processed-data-home  -p      TEXT  Processed data directory [required] │
  │ *  --model-home           -m      TEXT  Directory of the trained model to   │
  │                                         score with                          │
  │                                         [required]                          │
  │    --output-home          -o      TEXT  Output directory for scores,        │
  │                                         defaults to processed-data-home     │
  │    --verbose              -v            Verbose logging for collate         │
  │    --help                               Show this message and exit.         │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

[^1]:
    L. Gersony, "The Quiet Victory of Chicago’s Monk Parakeets," _The Chicago
    Maroon_, 23 January 2022,
    https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/

[^2]:
    M. Burkhart, B. Ramadan, Z. Liao, K. Chhikara, J. Rojas, W. Parker, & B.
    Beaulieu-Jones, Foundation models for electronic health records:
    representation dynamics and transferability,
    [arXiv:2504.10422](https://doi.org/10.48550/arXiv.2504.10422)

[^3]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188

[^4]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)

[^5]:
    I. Lee, L. Solo, M. Burkhart, B. Ramadan, W. Parker, & B. Beaulieu-Jones,
    Representation before training: a fixed-budget benchmark for generative
    medical event models,
    [arXiv:2604.16775](https://doi.org/10.48550/arXiv.2604.16775)

<!--

Run in tmux:
```
tmux new -s co || tmux a -t co
```

Format:
```sh
ruff format .
ruff check . --fix
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "output/" \
 --exclude "wandb/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cotorra \
 bbj-lab1:~
```

-->
