# Extraction

The extraction stage runs a trained model over timelines and captures the
representations it computes, so that a lightweight classifier can later be fit
on them (see [`RepBasedScorer`](scoring.md#repbasedscorer)). This is the
representation-based counterpart to autoregressive
[generative scoring](scoring.md#generativescorer).

## `Extractor`

`Extractor` loads a saved model, moves it to the best available device
(CUDA, MPS, or CPU), and runs it in inference mode over the held-out timelines.
For each timeline it reads out the model's last hidden-layer activations — by
default the single vector at the final token before the first `EOS` (the
model's summary of the timeline up to the prediction point), or, when
`all_times` is set, the full sequence of per-token representations padded to
`max_seq_len`. Results are written to `output_home` as sharded parquet files,
one column of feature vectors per timeline, ready to be consumed downstream.

::: cotorra.extractor.Extractor
