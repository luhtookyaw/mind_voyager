## Formula

Current difficulty index formula is:

$\text{difficulty\_index}=0.5 \cdot (1-\text{reveal\_rate})+0.3 \cdot \left(1-\min\left(\frac{\text{p90}}{\text{threshold}},1\right)\right)+0.2 \cdot \left(1-\min\left(\frac{\text{max}}{\text{threshold}},1\right)\right)$

In plain terms:

- `50%` weight: how rarely the field reveals
- `30%` weight: how far the `p90` score is below threshold
- `20%` weight: how far the `max` score is below threshold

Where:
- `reveal_rate = reveals / events`
- `p90 = 90th percentile of scored similarities`
- `max = maximum scored similarity`
- `threshold = current reveal threshold for that field`
- ratios are capped at `1.0`, so scores above threshold do not reduce difficulty below zero

So:
- higher difficulty index = harder to reveal
- lower difficulty index = easier to reveal

Interpretation:
- if reveal rate is low and scores are far below threshold, difficulty is high
- if reveal rate is high and high-percentile scores meet threshold, difficulty is low

Range:
- approximately `0` to `1`

- near `1`: very hard
- near `0`: easy

I chose `0.5 / 0.3 / 0.2` to make **actual reveal behavior** matter most, while still using score-distribution evidence to explain why.

So:

- `0.5` for `reveal_rate`
- `0.3` for `p90 / threshold`
- `0.2` for `max / threshold`

## Weight Justification

Reasoning:

**1. Reveal rate gets the largest weight (`0.5`)**
Because reveal is the actual outcome you care about.

If a field never reveals, that is the strongest signal that it is hard under the current setup, regardless of what the average score looks like.

Why not even larger, like `0.8`?
Because reveal rate alone can be misleading:
- maybe a field barely misses threshold many times
- maybe the threshold is slightly too strict
- maybe the field had only a few strong opportunities

So reveal rate should dominate, but not fully determine the result.

**2. `p90 / threshold` gets the next largest weight (`0.3`)**
Because `p90` tells you what strong but not extreme turns look like.

This is better than mean because:
- mean gets dragged down by many generic turns
- `p90` focuses on the better therapist questions
- it tells you whether the field is usually “within reach”

If `p90` is close to threshold:
- the field is probably not intrinsically hard
- the threshold may just be slightly too high

So `p90` helps distinguish:
- “hard because therapist never really gets close”
from
- “hard because threshold is a bit too strict”

**3. `max / threshold` gets the smallest weight (`0.2`)**
Because max is useful, but noisy.

A single unusually good question can produce a high max even if the field is generally hard.
So max is evidence of potential, but not stable evidence.

That is why it should matter less than:
- reveal rate
- p90

Still, it is worth keeping because:
- if even the max is far below threshold, the field is clearly too hard
- if max is above threshold but reveals are rare, maybe the field is reachable but inconsistent

So the weights reflect this trust order:

1. actual reveals
2. strong typical high-end behavior
3. best-case behavior

That is exactly:

- `0.5` reveal rate
- `0.3` p90
- `0.2` max

Why not use mean?
Because mean is much noisier for your setting:
- lots of generic therapist reflections lower the mean
- some fields get scored more often than others
- mean captures all turns, not the meaningful upper range

For threshold calibration, upper-tail behavior matters more than average behavior.

So the formula is designed to answer:

- Does this field actually reveal?
- If not, do strong questions usually get close?
- If not, does even the best question get close?

That is why the weighting is set that way.