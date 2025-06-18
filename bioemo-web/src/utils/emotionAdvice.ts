import type { EmotionPrediction, EmotionAdvice } from '../types';

const getEmotionAdvice = (emotion: string): EmotionAdvice => {
  switch (emotion.toLowerCase()) {
    case 'happiness':
      return {
        title: 'Keep up the positive energy!',
        description: 'Your positive emotional state contributes to a healthy work environment. Maintain this balance while staying mindful of your colleagues.',
        suggestions: [
          'Share your positive energy with team members',
          'Take on challenging tasks while in this productive state',
          'Document your successful strategies for future reference',
          'Consider mentoring or helping others who might need support'
        ]
      };
    case 'neutral':
      return {
        title: 'Maintaining a balanced perspective',
        description: 'Your composed state is ideal for focused work and rational decision-making. Use this clarity to tackle important tasks.',
        suggestions: [
          'Prioritize complex tasks requiring careful attention',
          'Schedule important meetings or discussions',
          'Review and plan your upcoming work objectives',
          'Take time for mindful reflection on your goals'
        ]
      };
    case 'sadness':
      return {
        title: 'Taking care of your wellbeing',
        description: 'It\'s normal to experience moments of sadness. Consider taking steps to support your emotional wellbeing.',
        suggestions: [
          'Take short breaks to reset and recharge',
          'Connect with supportive colleagues or your supervisor',
          'Focus on manageable tasks to maintain productivity',
          'Consider using available wellness resources'
        ]
      };
    case 'anger':
      return {
        title: 'Managing stress effectively',
        description: 'Your current state might benefit from stress management techniques. Take a moment to recenter yourself.',
        suggestions: [
          'Take a brief walk or practice deep breathing',
          'Postpone important decisions if possible',
          'Use stress management techniques',
          'Consider discussing concerns with appropriate channels'
        ]
      };
    case 'surprise':
      return {
        title: 'Adapting to unexpected situations',
        description: 'Surprise can be an opportunity for learning and growth. Use this heightened awareness to your advantage.',
        suggestions: [
          'Document any unexpected findings or situations',
          'Take time to assess and understand the situation',
          'Communicate important discoveries to relevant team members',
          'Update plans and strategies as needed'
        ]
      };
    case 'fear':
      return {
        title: 'Building confidence and security',
        description: 'Acknowledge your concerns while focusing on constructive actions. Remember that support is available.',
        suggestions: [
          'Break down challenging tasks into smaller steps',
          'Seek clarification on unclear expectations',
          'Connect with mentors or experienced colleagues',
          'Review and strengthen your security practices'
        ]
      };
    case 'disgust':
      return {
        title: 'Maintaining professional perspective',
        description: 'Channel your strong reactions into constructive improvements. Focus on solutions rather than problems.',
        suggestions: [
          'Document specific concerns objectively',
          'Propose constructive improvements',
          'Take breaks when needed to maintain composure',
          'Focus on tasks that align with your values'
        ]
      };
    default:
      return {
        title: 'Focusing on balanced productivity',
        description: 'Maintain a professional approach while being mindful of your emotional state.',
        suggestions: [
          'Focus on routine tasks that require attention',
          'Take regular breaks to maintain balance',
          'Stay connected with your team',
          'Use available resources when needed'
        ]
      };
  }
};

export const generateAdvice = (emotions: EmotionPrediction): EmotionAdvice => {
  // Find the dominant emotion
  const dominantEmotion = Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0];
  return getEmotionAdvice(dominantEmotion);
}; 