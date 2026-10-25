# Fine-tuning run report

Run: `20261025T181550Z_finetuning_e02bced7`

Official splits: 8,530 train / 1,066 validation / 1,066 test.

## Test metrics

```json
{
  "accuracy": 0.8395872420262664,
  "binary_precision": 0.8351851851851851,
  "binary_recall": 0.8461538461538461,
  "binary_f1": 0.8406337371854613,
  "macro_precision": 0.8396458245317561,
  "macro_recall": 0.8395872420262664,
  "macro_f1": 0.8395803246833822
}
```

## Qualitative error review

The examples below come from this run. Linguistic explanations are hypotheses, not measured model causes.

- Review: hey arnold ! is now stretched to barely feature length , with a little more attention paid to the animation . still , the updated dickensian sensibility of writer craig bartlett's story is appealing .
  True=0, predicted=1, confidence=0.996.
  Hypothesis: negation scope may reverse the sentiment of nearby words. This is a linguistic hypothesis, not a causal attribution.
- Review: entertains not so much because of its music or comic antics , but through the perverse pleasure of watching disney scrape the bottom of its own cracker barrel .
  True=0, predicted=1, confidence=0.996.
  Hypothesis: contrast or concession may mix positive and negative cues; negation scope may reverse the sentiment of nearby words. This is a linguistic hypothesis, not a causal attribution.
- Review: a movie where story is almost an afterthought amidst a swirl of colors and inexplicable events .
  True=1, predicted=0, confidence=0.996.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
- Review: the wild thornberrys movie has all the sibling rivalry and general family chaos to which anyone can relate .
  True=1, predicted=0, confidence=0.996.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
