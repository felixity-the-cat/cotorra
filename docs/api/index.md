# API Reference

Auto-generated documentation for the public `cotorra` API.

Cotorra trains generative event models on tokenized timelines and turns them
into predictions. The public API follows the same three-stage pipeline exposed
by the [CLI](../index.md#usage):

1. **[Training](training.md)** — fit a causal language model on packed token
   sequences ([`Trainer`](training.md#trainer)), optionally under differential
   privacy ([`TrainerDP`](training.md#trainerdp-differential-privacy)) or with
   hyperparameter search
   ([`Tuner`](training.md#tuner)).
2. **[Extraction](extraction.md)** — run a trained model over held-out
   timelines and dump its hidden-state representations
   ([`Extractor`](extraction.md#extractor)).
3. **[Scoring](scoring.md)** — turn a model into outcome predictions, either by
   autoregressive generation ([`GenerativeScorer`](scoring.md#generativescorer))
   or by fitting a classifier on the extracted representations
   ([`RepBasedScorer`](scoring.md#repbasedscorer)).

Every stage is driven by a YAML configuration file (with sensible packaged
defaults) and reads two artifacts produced upstream: a `tokens_times.parquet`
table of tokenized timelines and a `tokenizer.yaml` describing the vocabulary.
See [Context](../index.md#context) and [Configuration](../index.md#configuration)
for the expected inputs.
