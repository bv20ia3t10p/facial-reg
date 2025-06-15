import type { EmotionPrediction } from '../types';

interface EmotionAdvice {
  title: string;
  description: string;
  suggestions: string[];
}

const emotionAdviceMap: Record<string, EmotionAdvice> = {
  happiness: {
    title: 'Keep up the positive energy!',
    description: 'Your positive emotional state contributes to a better work environment.',
    suggestions: [
      'Share your positive energy with colleagues',
      'Take on new challenges while in this positive mindset',
      'Document your achievements and progress',
    ],
  },
  neutral: {
    title: 'Maintaining Balance',
    description: 'You\'re in a balanced emotional state, which is great for focused work.',
    suggestions: [
      'Use this balanced state for complex problem-solving',
      'Plan your upcoming tasks and priorities',
      'Consider learning something new',
    ],
  },
  sadness: {
    title: 'Taking Care of Your Wellbeing',
    description: 'It\'s okay to feel down sometimes. Consider taking steps to lift your mood.',
    suggestions: [
      'Take short breaks to refresh your mind',
      'Connect with supportive colleagues',
      'Focus on manageable tasks to build momentum',
    ],
  },
  anger: {
    title: 'Managing Stress',
    description: 'Let\'s channel this energy constructively.',
    suggestions: [
      'Take a brief walk or breathing exercise',
      'Write down your concerns to address them systematically',
      'Consider discussing issues with your supervisor or HR',
    ],
  },
  fear: {
    title: 'Building Confidence',
    description: 'Let\'s work on addressing your concerns.',
    suggestions: [
      'Break down challenging tasks into smaller steps',
      'Seek clarification on unclear expectations',
      'Connect with a mentor or experienced colleague',
    ],
  },
  surprise: {
    title: 'Adapting to Change',
    description: 'Change can be an opportunity for growth.',
    suggestions: [
      'Take a moment to process new information',
      'List questions and seek clarification',
      'Share your insights with the team',
    ],
  },
  disgust: {
    title: 'Addressing Concerns',
    description: 'Let\'s work on improving your work environment.',
    suggestions: [
      'Identify specific issues that need attention',
      'Discuss concerns with appropriate channels',
      'Focus on aspects of work you find meaningful',
    ],
  },
};

export function generateAdvice(emotions: EmotionPrediction): EmotionAdvice {
  // Find the dominant emotion
  const dominantEmotion = Object.entries(emotions).reduce((a, b) => 
    a[1] > b[1] ? a : b
  )[0];

  return emotionAdviceMap[dominantEmotion] || emotionAdviceMap.neutral;
}

export function getDominantEmotion(emotions: EmotionPrediction): string {
  return Object.entries(emotions).reduce((a, b) => 
    a[1] > b[1] ? a : b
  )[0];
} 