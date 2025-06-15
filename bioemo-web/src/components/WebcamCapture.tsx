import React, { useCallback, useRef } from 'react';
import Webcam from 'react-webcam';
import { LoadingOutlined, CameraOutlined } from '@ant-design/icons';
import { Typography } from 'antd';

const { Text } = Typography;

interface WebcamCaptureProps {
  onCapture: (imageSrc: string) => void;
  isScanning: boolean;
}

export function WebcamCapture({ onCapture, isScanning }: WebcamCaptureProps) {
  const webcamRef = useRef<Webcam>(null);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      onCapture(imageSrc);
    }
  }, [onCapture]);

  return (
    <div style={{
      width: '100%',
      aspectRatio: '16/9',
      backgroundColor: '#f0f2f5',
      borderRadius: '8px',
      overflow: 'hidden',
      position: 'relative',
    }}>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
        }}
      />
      
      {isScanning && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
        }}>
          <LoadingOutlined style={{ fontSize: 48, color: '#1890ff' }} spin />
        </div>
      )}

      {!webcamRef.current && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
        }}>
          <CameraOutlined style={{ fontSize: 48, color: '#8c8c8c' }} />
          <Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
            Camera feed will appear here
          </Text>
        </div>
      )}
    </div>
  );
} 