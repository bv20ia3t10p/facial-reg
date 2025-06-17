import type { EmotionPrediction } from '../types';

interface EmotionAdvice {
  title: string;
  description: string;
  suggestions: string[];
}

const emotionAdviceMap: Record<string, EmotionAdvice> = {
  happiness: {
    title: 'Thriving at Work! 🌟',
    description: 'Your positive emotional state is a valuable asset to the team. Use this energy to boost productivity and inspire others.',
    suggestions: [
      'Take on challenging projects that require creativity and innovation',
      'Mentor or support team members who might benefit from your positive energy',
      'Document your successful strategies for future reference',
      'Set ambitious goals while maintaining this positive momentum',
      'Share your achievements and celebrate team successes'
    ],
  },
  neutral: {
    title: 'Balanced and Focused 🎯',
    description: 'Your balanced state is ideal for analytical tasks and strategic thinking. Make the most of this clear-headed mindset.',
    suggestions: [
      'Tackle complex problems that require careful analysis',
      'Review and optimize your work processes',
      'Plan upcoming projects and set clear milestones',
      'Engage in strategic discussions with team members',
      'Update your skills through focused learning sessions'
    ],
  },
  sadness: {
    title: 'Taking Care of Your Wellbeing 💙',
    description: 'Everyone experiences down moments. Let\'s focus on self-care and gradual improvement while maintaining professional responsibilities.',
    suggestions: [
      'Break your workday into smaller, manageable chunks',
      'Schedule a casual chat with a trusted colleague or mentor',
      'Take regular short breaks for physical movement or fresh air',
      'Focus on tasks that give you a sense of accomplishment',
      'Consider speaking with HR about wellness resources'
    ],
  },
  anger: {
    title: 'Channeling Energy Productively ⚡',
    description: 'Strong emotions can be transformed into productive action. Let\'s focus on constructive solutions and professional growth.',
    suggestions: [
      'Take a 5-minute mindfulness break to center yourself',
      'Document your concerns clearly and objectively',
      'Schedule a structured discussion with relevant stakeholders',
      'Focus on process improvements that could prevent future issues',
      'Engage in physical activity during your break to release tension'
    ],
  },
  fear: {
    title: 'Building Confidence and Clarity 🌅',
    description: 'Uncertainty is a natural part of growth. Let\'s break down challenges into manageable steps and build confidence gradually.',
    suggestions: [
      'Create a detailed action plan for challenging tasks',
      'Schedule a meeting with your supervisor for clear guidance',
      'Practice new skills in a low-pressure environment',
      'Connect with colleagues who have overcome similar challenges',
      'Document your progress to visualize your growth'
    ],
  },
  surprise: {
    title: 'Embracing Change and Growth 🚀',
    description: 'Unexpected situations can open doors to new opportunities. Let\'s channel this energy into productive adaptation.',
    suggestions: [
      'Document new developments and their potential impact',
      'Identify areas where change could lead to improvement',
      'Share your fresh perspectives with the team',
      'Create an action plan to adapt to new circumstances',
      'Look for learning opportunities in the unexpected'
    ],
  },
  disgust: {
    title: 'Improving Your Work Environment 🌿',
    description: 'Your high standards can drive positive change. Let\'s focus on constructive improvements while maintaining professionalism.',
    suggestions: [
      'Create a detailed list of specific issues that need addressing',
      'Develop practical solutions to propose to management',
      'Focus on aspects of your work that align with your values',
      'Seek out projects that match your quality standards',
      'Connect with colleagues who share your commitment to excellence'
    ],
  },
};

export function generateAdvice(emotions: EmotionPrediction): EmotionAdvice {
  // Find the dominant emotion
  const dominantEmotion = getDominantEmotion(emotions);
  
  // Get the base advice for the dominant emotion
  const baseAdvice = emotionAdviceMap[dominantEmotion] || emotionAdviceMap.neutral;
  
  // Check for secondary emotions that might influence the advice
  const sortedEmotions = Object.entries(emotions)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 2); // Get top 2 emotions
  
  if (sortedEmotions.length > 1) {
    const [primary, secondary] = sortedEmotions;
    const [primaryEmotion, primaryValue] = primary;
    const [secondaryEmotion, secondaryValue] = secondary;
    
    // If secondary emotion is significant (>30% of primary)
    if (secondaryValue > primaryValue * 0.3) {
      // Add a suggestion from the secondary emotion
      const secondaryAdvice = emotionAdviceMap[secondaryEmotion];
      if (secondaryAdvice) {
        baseAdvice.suggestions = [
          ...baseAdvice.suggestions.slice(0, 4),
          secondaryAdvice.suggestions[0]
        ];
      }
    }
  }
  
  return baseAdvice;
}

export function getDominantEmotion(emotions: EmotionPrediction): string {
  return Object.entries(emotions).reduce((a, b) => 
    a[1] > b[1] ? a : b
  )[0];
} 