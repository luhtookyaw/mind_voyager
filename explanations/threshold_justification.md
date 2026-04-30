## Threshold Justification

The justification is that difficulty should reflect **how strong a therapist turn must be, relative to the observed score distribution**, not an arbitrary fixed number.

So:

- `easy = P75`
- `normal = (P75 + P90)/2`
- `hard = P90`

means:

- `easy`: a therapist turn only needs to be in the upper-middle range of observed turns
- `normal`: a therapist turn needs to be clearly above average and close to the stronger end
- `hard`: a therapist turn needs to be among the strongest observed turns

Why this makes sense:

**Easy = P75**
- A turn at the 75th percentile is better than most turns, but not exceptional
- That matches the idea of easy difficulty:
  - a reasonably good question should often unlock disclosure
- It avoids making easy trivial, because not every average question will pass

**Normal = (P75 + P90)/2**
- This creates an intermediate standard between “good” and “strong”
- It matches the idea of normal difficulty:
  - the therapist needs a clearly targeted question, not just a decent one
- It also gives smoother spacing between easy and hard

**Hard = P90**
- A turn at the 90th percentile is near the top end of observed performance
- That matches the idea of hard difficulty:
  - only strong, semantically precise probing should unlock disclosure
- It makes hard selective without requiring a completely outlier turn

So the conceptual basis is:

- difficulty is defined by **percentile rank within the empirical turn-quality distribution**
- not by absolute cosine values alone

That is a defensible justification because it ties the thresholds to:
- actual observed therapist behavior
- relative semantic strength
- graded disclosure standards

A compact way to write it in a method section:

- `We define reveal thresholds by empirical similarity percentiles so that easy, normal, and hard correspond to increasingly strong therapist prompts relative to the observed score distribution. Easy uses the 75th percentile, hard uses the 90th percentile, and normal uses the midpoint between them.`

Why this is better than using mean:
- percentiles are more robust to many low-quality generic turns
- they reflect the upper part of the distribution, which is what matters for elicitation

Why this is better than using max:
- max depends on one outlier turn
- percentiles are much more stable

The main caveat:
- this justification is strongest if the distribution is computed from a representative set of sessions
- and if you accept that thresholds are relative to the therapist policy used during calibration

So the justification is:
- `easy` = reasonably strong turn
- `normal` = clearly strong turn
- `hard` = top-end turn

That is coherent, empirical, and easy to explain.