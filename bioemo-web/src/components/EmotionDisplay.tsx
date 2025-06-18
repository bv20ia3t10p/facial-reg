import React from 'react';
import { Space, Tooltip, Progress, Typography } from 'antd';
import { SmileOutlined, SafetyOutlined, HeartOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { emotionColors } from '../constants/colors';
import type { EmotionPrediction } from '../types';

const { Text } = Typography;

const emotionDisplayMap: Record<keyof EmotionPrediction, string> = {
  neutral: 'Neutral',
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
  surprised: 'Surprised',
  fearful: 'Fearful',
  disgusted: 'Disgusted'
};

export const EmotionIcon = ({ emotion }: { emotion: string }) => {
  const iconStyle = { fontSize: '24px' };
  switch (emotion.toLowerCase()) {
    case 'happy': return <SmileOutlined style={iconStyle} />;
    case 'neutral': return <SafetyOutlined style={iconStyle} />;
    case 'sad': return <HeartOutlined style={iconStyle} />;
    case 'angry': return <ThunderboltOutlined style={iconStyle} />;
    default: return <SmileOutlined style={iconStyle} />;
  }
};

interface EmotionDisplayProps {
  emotions: EmotionPrediction;
  compact?: boolean;
}

export const EmotionDisplay: React.FC<EmotionDisplayProps> = ({ emotions, compact = true }) => {
  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      {Object.entries(emotions).map(([emotion, value]) => (
        <div key={emotion}>
          <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: compact ? 4 : 8 }}>
            <Text strong style={{ 
              textTransform: 'capitalize', 
              fontSize: compact ? '14px' : '16px',
              color: emotionColors[emotion] || '#1677ff'
            }}>
              <EmotionIcon emotion={emotion} /> {emotionDisplayMap[emotion as keyof EmotionPrediction] || emotion}
            </Text>
            <Text>{(value * 100).toFixed(1)}%</Text>
          </Space>
          <Tooltip title={`${(value * 100).toFixed(1)}% ${emotionDisplayMap[emotion as keyof EmotionPrediction] || emotion}`}>
            <Progress 
              percent={value * 100} 
              showInfo={false}
              strokeColor={{
                '0%': emotionColors[emotion] || '#1677ff',
                '100%': `${emotionColors[emotion]}90` || '#1677ff90'
              }}
              strokeLinecap="round"
              style={{ 
                marginBottom: compact ? '8px' : '16px', 
                height: compact ? 8 : 12,
                background: `${emotionColors[emotion]}10` || '#1677ff10'
              }}
            />
          </Tooltip>
        </div>
      ))}
    </Space>
  );
}; 