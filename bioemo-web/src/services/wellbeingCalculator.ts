import type { EmotionPrediction } from '../types';

/**
 * Weights for different emotions in stress level calculation
 * Higher weights for negative emotions indicate more stress
 */
const STRESS_WEIGHTS = {
  anger: 0.35,    // High impact on stress
  fear: 0.25,     // Significant impact
  disgust: 0.15,  // Moderate impact
  sadness: 0.15,  // Moderate impact
  surprise: 0.05, // Low impact
  neutral: -0.05, // Slightly reduces stress
  happiness: -0.1 // Reduces stress
};

/**
 * Weights for different emotions in job satisfaction calculation
 * Positive emotions have higher weights
 */
const SATISFACTION_WEIGHTS = {
  happiness: 0.4,  // Strong indicator of satisfaction
  neutral: 0.1,    // Mild positive indicator
  surprise: 0.05,  // Slight positive
  sadness: -0.2,   // Negative indicator
  disgust: -0.15,  // Negative indicator
  anger: -0.25,    // Strong negative indicator
  fear: -0.15      // Negative indicator
};

/**
 * Weights for emotional balance calculation
 * Based on the stability and positivity of emotions
 */
const BALANCE_WEIGHTS = {
  neutral: 0.3,     // High weight for stability
  happiness: 0.25,  // Positive emotion
  surprise: 0.1,    // Mild impact
  sadness: -0.15,   // Negative impact
  disgust: -0.15,   // Negative impact
  anger: -0.2,      // Strong negative impact
  fear: -0.15       // Negative impact
};

/**
 * Calculates stress level based on emotion distribution
 * @param emotions Object containing emotion probabilities
 * @returns number between 0-100, where higher numbers indicate more stress
 */
export function calculateStressLevel(emotions: EmotionPrediction): number {
  let stressScore = 0;
  let totalWeight = 0;

  Object.entries(emotions).forEach(([emotion, probability]) => {
    if (emotion in STRESS_WEIGHTS) {
      stressScore += probability * STRESS_WEIGHTS[emotion as keyof typeof STRESS_WEIGHTS];
      totalWeight += Math.abs(STRESS_WEIGHTS[emotion as keyof typeof STRESS_WEIGHTS]);
    }
  });

  // Normalize to 0-100 scale and invert (higher stress = higher number)
  const normalizedScore = ((stressScore / totalWeight) + 1) * 50;
  return Math.min(Math.max(normalizedScore, 0), 100);
}

/**
 * Calculates job satisfaction based on emotion distribution
 * @param emotions Object containing emotion probabilities
 * @returns number between 0-100, where higher numbers indicate more satisfaction
 */
export function calculateJobSatisfaction(emotions: EmotionPrediction): number {
  let satisfactionScore = 0;
  let totalWeight = 0;

  Object.entries(emotions).forEach(([emotion, probability]) => {
    if (emotion in SATISFACTION_WEIGHTS) {
      satisfactionScore += probability * SATISFACTION_WEIGHTS[emotion as keyof typeof SATISFACTION_WEIGHTS];
      totalWeight += Math.abs(SATISFACTION_WEIGHTS[emotion as keyof typeof SATISFACTION_WEIGHTS]);
    }
  });

  // Normalize to 0-100 scale
  const normalizedScore = ((satisfactionScore / totalWeight) + 1) * 50;
  return Math.min(Math.max(normalizedScore, 0), 100);
}

/**
 * Calculates emotional balance based on emotion distribution
 * @param emotions Object containing emotion probabilities
 * @returns number between 0-100, where higher numbers indicate better balance
 */
export function calculateEmotionalBalance(emotions: EmotionPrediction): number {
  let balanceScore = 0;
  let totalWeight = 0;

  Object.entries(emotions).forEach(([emotion, probability]) => {
    if (emotion in BALANCE_WEIGHTS) {
      balanceScore += probability * BALANCE_WEIGHTS[emotion as keyof typeof BALANCE_WEIGHTS];
      totalWeight += Math.abs(BALANCE_WEIGHTS[emotion as keyof typeof BALANCE_WEIGHTS]);
    }
  });

  // Calculate emotion diversity (Shannon entropy)
  const entropy = Object.values(emotions).reduce((acc, p) => {
    if (p > 0) {
      return acc - (p * Math.log2(p));
    }
    return acc;
  }, 0);

  // Combine balance score with entropy
  const normalizedScore = ((balanceScore / totalWeight) + 1) * 40;
  const entropyContribution = (entropy / Math.log2(Object.keys(emotions).length)) * 20;
  
  return Math.min(Math.max(normalizedScore + entropyContribution, 0), 100);
}

/**
 * Calculates overall wellbeing score based on other metrics
 * @param stressLevel Stress level (0-100)
 * @param jobSatisfaction Job satisfaction (0-100)
 * @param emotionalBalance Emotional balance (0-100)
 * @returns number between 0-100, where higher numbers indicate better wellbeing
 */
export function calculateWellbeingScore(
  stressLevel: number,
  jobSatisfaction: number,
  emotionalBalance: number
): number {
  // Invert stress level (higher stress = lower wellbeing)
  const stressContribution = (100 - stressLevel) * 0.3;  // 30% weight
  const satisfactionContribution = jobSatisfaction * 0.4; // 40% weight
  const balanceContribution = emotionalBalance * 0.3;     // 30% weight

  return Math.round(
    stressContribution + satisfactionContribution + balanceContribution
  );
}

/**
 * Calculates all wellbeing metrics from emotion data
 * @param emotions Object containing emotion probabilities
 * @returns Object containing all wellbeing metrics
 */
export function calculateAllMetrics(emotions: EmotionPrediction) {
  const stressLevel = calculateStressLevel(emotions);
  const jobSatisfaction = calculateJobSatisfaction(emotions);
  const emotionalBalance = calculateEmotionalBalance(emotions);
  const wellbeingScore = calculateWellbeingScore(
    stressLevel,
    jobSatisfaction,
    emotionalBalance
  );

  return {
    stressLevel: Math.round(stressLevel),
    jobSatisfaction: Math.round(jobSatisfaction),
    emotionalBalance: Math.round(emotionalBalance),
    wellbeingScore: Math.round(wellbeingScore)
  };
} 