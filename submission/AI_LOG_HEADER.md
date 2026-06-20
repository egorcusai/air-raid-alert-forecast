# AI Interaction Log — Reading Guide

**Project:** Air-Raid Alert Risk Classification (Ukraine)
**Repo:** https://github.com/egorcusai/air-raid-alert-forecast
**Tool used:** Claude (Anthropic)

---

## How to read this log

This log captures the *real* engineering process — including the wrong turns and
their corrections, which are the point. The conversation was a genuine
back-and-forth: I directed, challenged, and corrected the model rather than
accepting first outputs. The five moments below are where that iteration is
most visible.

### Key iteration moments (where to look)

1. **Data-source verification, not assumption.** Rather than letting the model
   emit a dataset URL from memory, I had it *search and verify* the real source,
   then inspect the actual schema. This surfaced two landmines a naive approach
   would have missed: the dataset switched from oblast- to raion-level in Dec
   2025, and two regions have permanent unlogged sirens. Both were handled
   explicitly.

2. **Architecture pivot.** The first design (exact-time forecasting, random
   train/test split) was challenged and replaced with 3-hour-ahead binary
   classification and a strict chronological split — because random splits leak
   the future in a time series.

3. **Catching a second, subtler leak mid-build.** After the first model ran, we
   found the decision threshold was being tuned on the *test* set — a leak of a
   different kind. It was moved to a separate validation block. This correction
   is the clearest example of not accepting a working-but-flawed output.

4. **Hypothesis → test → honest result.** The initial model was near-random
   (ROC-AUC ~0.54). Rather than hide this, I hypothesised the missing signal was
   geographic propagation, added neighbouring-region features, and tested across
   ten regions. The finding — western/peripheral regions are predictable, direct
   targets are not — reproduces a published result and is reported honestly.

5. **Risk/calibration honesty.** A correlation matrix quantified the propagation
   thesis (adjacent western oblasts correlate 0.84; distant pairs ~0.12), and a
   calibration check (ECE ~0.25) led to the honest caveat that the model's
   probabilities are a relative *ranking*, not absolute likelihoods.

### What this log demonstrates
Iterating, challenging model outputs, correcting errors (two distinct data-leak
fixes), and steering toward a defensible architecture — over chasing an inflated
accuracy number.
