1. Data setup

For each concept, we already have:

Positive pole examples (very “+X”)

Negative pole examples (very “–X”)

Neutral band (things that are neither strongly + nor −, or explicitly “neutral / mixed / irrelevant”)

Use those to build three sets:

D_pos_extreme: positive, high confidence

D_neg_extreme: negative, high confidence

D_neutral: central / neutral examples

2. Head A: Sign head (positive vs negative)

This head answers:

“If this is an extreme instance of the concept, which side is it on?”

Training data: only D_pos_extreme ∪ D_neg_extreme

Label y_sign = 1 for positive, 0 for negative.

Drop neutrals entirely for this head.

Model: simple linear lens on hidden states, or the same direction we already use:

ssign=σ(wsign⊤h+bsign)


Loss: standard binary cross-entropy on those extreme examples.

So this head purely learns the direction of the axis: which side is + vs −.

3. Head B: Extremeness head (extreme vs neutral)

This head answers:

“Is this an extreme instance of the concept, or basically neutral?”

Training data:

D_pos_extreme → label 1

D_neg_extreme → label 1

D_neutral → label 0

Model: another linear head (can share the same underlying features / direction):

sext=σ(wext⊤h+bext)



Loss: binary cross-entropy on all three groups.

So this head learns something close to |projection| along the concept axis (or at least: “far from centre vs around centre”), independent of sign.

4. Joint training

we just sum the two losses:

L=λsignLsign+λextLext


where

L_sign is BCE over extremes (pos/neg)

L_ext is BCE over extremes + neutrals (extreme vs neutral)

we can re-use the same representation and even the same underlying direction if we want:

e.g. use one projection z = vᵀh, then:

sign head = σ(a·z + b)

extremeness head = σ(c·|z| + d) or σ(c·z² + d)

…but keeping two separate linear heads is simpler and more flexible.

5. Inference: turn it into three poles

At inference, we combine the two signals:

Let:

p_sign = s_sign(h) (probability it’s positive)

p_ext = s_ext(h) (probability it’s extreme vs neutral)

Choose thresholds (after calibration, e.g. Platt / isotonic):

τ_ext for “extreme”

optionally τ_pos / τ_neg for sign confidence

Then:

Neutral:

If p_ext < τ_ext → Neutral pole

Positive pole:

If p_ext ≥ τ_ext and p_sign ≥ 0.5 → Positive

Negative pole:

If p_ext ≥ τ_ext and p_sign < 0.5 → Negative
we can also output a soft 3-vector for downstream steering:

P(neutral) = 1 − p_ext

P(positive) = p_ext · p_sign

P(negative) = p_ext · (1 − p_sign)

That’s our three-pole classifier without ever training a single 3-way softmax head.