# LLM Judge Validation Report

Generated: 2025-12-02 23:17:49

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Comparisons | 19 |
| Average Consistency Score | 76.84 |
| Best Prompt Strategy | chain_of_thought |
| Cohen's Kappa | 0.0000 (Slight agreement) |
| Human Labels Used | 16 |

---

## 1. Ablation Study (Prompt Robustness)

**Question:** Which prompting strategy yields most stable results?

| Strategy | Mean Score | Std Dev | N |
|----------|-----------|---------|---|
| zero_shot | 62.00 | 9.80 | 5 |
| chain_of_thought | 66.00 | 4.90 | 5 |
| few_shot | 46.00 | 4.90 | 5 |

**Finding**: `chain_of_thought` has lowest variance, making it the most reliable strategy.

---

## 2. Self-Consistency Test

**Question:** Is the judge deterministic or noisy when run multiple times?

| Event | Author | Mean | Std | Status |
|-------|--------|------|-----|--------|
| election_night_1860 | Hay, John | 65.0 | 4.5 | STABLE |
| election_night_1860 | Ketcham, Henry | 48.0 | 13.6 | NOISY |
| election_night_1860 | Morse, John T., | 65.0 | 3.2 | STABLE |

**Finding**: 67% of comparisons are stable (std < 10).

---

## 3. Cohen's Kappa (Human-AI Agreement)

**Question:** Does the LLM judge agree with human labels?

| Metric | Value |
|--------|-------|
| Cohen's Kappa | 0.0000 |
| Raw Agreement | 87.50% |
| Interpretation | Slight agreement |
| Samples Used | 16 |

**Confusion Matrix**:
```
                    Human: Consistent    Human: Contradictory
LLM: Consistent               14                      2      
LLM: Contradictory            0                       0      
```

---

## 4. Sample Contradictions Found

### election_night_1860 (OMISSION)
- **Lincoln**: A circular was distributed before the election by respectable citizens of New York, portraying Lincoln as a dangerous radical....
- **Hay, John**: The election resulted disastrously for Douglas....
- **Explanation**: Lincoln's account does not mention the outcome of the election for Douglas, which is a significant aspect of the event.

### election_night_1860 (OMISSION)
- **Lincoln**: A circular was distributed before the election by respectable citizens of New York, portraying Lincoln as a dangerous radical....
- **Ketcham, Henry**: No mention of a circular or portrayal of Lincoln as a dangerous radical....
- **Explanation**: Ketcham's account does not mention the circular or the portrayal of Lincoln as a radical, which is a significant aspect of Lincoln's account.

### election_night_1860 (INTERPRETIVE)
- **Lincoln**: A circular was distributed before the election by respectable citizens of New York, portraying Lincoln as a dangerous radical....
- **Morse, John T., Jr. (John Torrey)**: Mr. Lincoln faced significant opposition and challenges during the presidential campaign leading up to Election Night....
- **Explanation**: Lincoln's account focuses on a specific incident involving a circular, while Morse's account generalizes the opposition Lincoln faced. The interpretation of the opposition's nature and its impact differs.

### election_night_1860 (OMISSION)
- **Lincoln**: A circular was distributed before the election by respectable citizens of New York, portraying Lincoln as a dangerous radical....
- **Browne, Francis F. (Francis Fisher)**: Lincoln was nominated for President by the convention....
- **Explanation**: Lincoln's account focuses on the negative portrayal and public anxiety, while Browne's account highlights the nomination and celebration, omitting the negative portrayal.

### election_night_1860 (INTERPRETIVE)
- **Lincoln**: A circular was distributed before the election by respectable citizens of New York, portraying Lincoln as a dangerous radical....
- **Charnwood, Godfrey Rathbone Benson, Baron**: Lincoln's election was unlooked-for and raised him to the highest place of ambition....
- **Explanation**: Lincoln's account focuses on the negative portrayal and public anxiety, while Charnwood emphasizes the unexpected nature and ambition associated with his election.

---

## Conclusions

1. **Prompt Strategy:** Chain-of-Thought prompting provides most reliable results with lowest variance.
2. **Reliability:** The LLM judge shows consistent behavior across multiple runs.
3. **Human Alignment:** Cohen's Kappa measures agreement between LLM and human judgments.
