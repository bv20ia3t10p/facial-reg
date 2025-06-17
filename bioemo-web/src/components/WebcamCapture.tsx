import React, { useCallback, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { LoadingOutlined, CameraOutlined } from '@ant-design/icons';
import { Typography, Button } from 'antd';

const { Text } = Typography;

interface WebcamCaptureProps {
  onCapture: (imageSrc: string) => void;
  isScanning: boolean;
}

export function WebcamCapture({ onCapture, isScanning }: WebcamCaptureProps) {
  const webcamRef = useRef<Webcam>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
  };

  const capture = useCallback(() => {
    if (!webcamRef.current) {
      const err = 'Webcam not initialized';
      console.error(err);
      setError(err);
      return;
    }

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        // Debug image data
        console.log('Image captured successfully');
        console.log('Image data length:', imageSrc.length);
        console.log('Image data starts with:', imageSrc.substring(0, 50));
        console.log('Image format:', imageSrc.split(';')[0]);
        
        // Verify image data is valid base64
        try {
          const base64Data = imageSrc.split(',')[1];
          atob(base64Data);
          console.log('Image is valid base64');
          
          // Create a test Blob to verify data
          const byteString = atob(base64Data);
          const mimeString = imageSrc.split(':')[1].split(';')[0];
          console.log('MIME type:', mimeString);
          
          const ab = new ArrayBuffer(byteString.length);
          const ia = new Uint8Array(ab);
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
          }
          const blob = new Blob([ab], { type: mimeString });
          console.log('Successfully created Blob:', blob.size, 'bytes');
        } catch (e) {
          console.error('Invalid base64 data:', e);
          setError('Invalid image data format');
          return;
        }
        
        onCapture(imageSrc);
        setError(null);
      } else {
        const err = 'Failed to capture image - no data returned';
        console.error(err);
        setError(err);
      }
    } catch (e) {
      const err = `Error capturing image: ${e}`;
      console.error(err);
      setError(err);
    }
  }, [onCapture]);

  const handleUserMedia = useCallback(() => {
    console.log('Camera is ready');
    console.log('Video constraints:', videoConstraints);
    setIsCameraReady(true);
    setError(null);
  }, []);

  const handleUserMediaError = useCallback((err: string | DOMException) => {
    const errorMessage = err instanceof DOMException ? err.message : err;
    console.error('Camera error:', errorMessage);
    setError(`Camera error: ${errorMessage}`);
    setIsCameraReady(false);
  }, []);

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
        onUserMedia={handleUserMedia}
        onUserMediaError={handleUserMediaError}
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

      {!isCameraReady && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
        }}>
          <CameraOutlined style={{ fontSize: 48, color: '#8c8c8c' }} />
          <Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
            {error || 'Camera feed will appear here'}
          </Text>
        </div>
      )}

      {isCameraReady && !isScanning && (
        <div style={{
          position: 'absolute',
          bottom: '16px',
          left: '50%',
          transform: 'translateX(-50%)',
        }}>
          <Button
            type="primary"
            icon={<CameraOutlined />}
            onClick={capture}
            size="large"
            style={{
              height: '48px',
              width: '48px',
              borderRadius: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          />
          {error && (
            <Text type="danger" style={{ display: 'block', marginTop: '8px', textAlign: 'center' }}>
              {error}
            </Text>
          )}
        </div>
      )}
    </div>
  );
} 