# Wellbeing Metrics Calculation Documentation

This document explains how various wellbeing metrics are calculated from emotional data in the BioEmo system.

## Overview

The system calculates four main metrics:
1. Stress Level
2. Job Satisfaction
3. Emotional Balance
4. Overall Wellbeing Score

All metrics are normalized to a 0-100 scale, where higher numbers generally indicate better wellbeing (except for stress level, where higher numbers indicate more stress).

## Input Data

The calculations use emotion probability distributions from facial recognition:
- happiness
- neutral
- surprise
- sadness
- anger
- disgust
- fear

Each emotion has a probability value between 0 and 1, with the sum of all probabilities equaling 1.

## Metric Calculations

### 1. Stress Level (0-100)

Stress level is calculated using weighted contributions from different emotions:

```
Emotion Weights:
- anger:    +0.35 (High impact)
- fear:     +0.25 (Significant impact)
- disgust:  +0.15 (Moderate impact)
- sadness:  +0.15 (Moderate impact)
- surprise: +0.05 (Low impact)
- neutral:  -0.05 (Slightly reduces stress)
- happiness:-0.10 (Reduces stress)
```

Calculation steps:
1. Multiply each emotion's probability by its weight
2. Sum all weighted values
3. Normalize to 0-100 scale
4. Higher values indicate more stress

Example:
```
If emotions = {
  happiness: 0.6,
  neutral: 0.3,
  anger: 0.1
}

Score = (0.6 * -0.10) + (0.3 * -0.05) + (0.1 * 0.35)
Normalized to 0-100 scale
```

### 2. Job Satisfaction (0-100)

Job satisfaction emphasizes positive emotions with these weights:

```
Emotion Weights:
- happiness: +0.40 (Strong positive)
- neutral:   +0.10 (Mild positive)
- surprise:  +0.05 (Slight positive)
- sadness:   -0.20 (Negative)
- disgust:   -0.15 (Negative)
- anger:     -0.25 (Strong negative)
- fear:      -0.15 (Negative)
```

Calculation steps:
1. Multiply each emotion's probability by its satisfaction weight
2. Sum all weighted values
3. Normalize to 0-100 scale
4. Higher values indicate more satisfaction

### 3. Emotional Balance (0-100)

Emotional balance combines two factors:
1. Weighted emotion scores
2. Emotional diversity (Shannon entropy)

```
Balance Weights:
- neutral:   +0.30 (High stability)
- happiness: +0.25 (Positive)
- surprise:  +0.10 (Mild impact)
- sadness:   -0.15 (Negative)
- disgust:   -0.15 (Negative)
- anger:     -0.20 (Strong negative)
- fear:      -0.15 (Negative)
```

Calculation steps:
1. Calculate weighted emotion score (40% of final score)
2. Calculate Shannon entropy of emotion distribution (20% of final score)
3. Combine and normalize to 0-100 scale

The entropy component rewards emotional diversity within healthy bounds:
```
Entropy = -∑(p * log2(p)) for each emotion probability p
```

### 4. Overall Wellbeing Score (0-100)

The overall wellbeing score combines the other metrics with these weights:
- Job Satisfaction: 40%
- Emotional Balance: 30%
- Stress Level: 30% (inverted, as lower stress = better wellbeing)

```
Wellbeing = (JobSatisfaction * 0.4) + 
            (EmotionalBalance * 0.3) + 
            ((100 - StressLevel) * 0.3)
```

## Implementation Notes

1. All calculations handle missing emotions gracefully
2. Results are rounded to nearest integer
3. Values are clamped to 0-100 range
4. Weights are based on psychological research and can be adjusted
5. Shannon entropy helps identify emotional numbness or volatility

## Usage Example

```typescript
const emotions = {
  happiness: 0.5,
  neutral: 0.3,
  anger: 0.1,
  sadness: 0.1
};

const metrics = calculateAllMetrics(emotions);
// Returns:
// {
//   stressLevel: 35,
//   jobSatisfaction: 78,
//   emotionalBalance: 82,
//   wellbeingScore: 75
// }
```

## References

The weighting systems are based on research in:
- Workplace stress indicators
- Job satisfaction measurement
- Emotional intelligence assessment
- Psychological wellbeing metrics

The calculations prioritize:
1. Long-term stability over short-term happiness
2. Balanced emotional states over extreme positivity
3. Sustainable workplace satisfaction
4. Early detection of stress patterns 