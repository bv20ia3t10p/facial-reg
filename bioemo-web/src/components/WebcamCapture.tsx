import React, { useCallback, useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import { LoadingOutlined, CameraOutlined, CheckOutlined } from '@ant-design/icons';
import { Typography, Button, Spin } from 'antd';
import { useTheme } from '../contexts/ThemeContext';

const { Text } = Typography;

interface WebcamCaptureProps {
  onCapture: (imageSrc: string) => void;
  isScanning: boolean;
}

export function WebcamCapture({ onCapture, isScanning }: WebcamCaptureProps) {
  const webcamRef = useRef<Webcam>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [showFaceScan, setShowFaceScan] = useState(false);
  const { isDarkMode } = useTheme();

  // Set video constraints to maintain aspect ratio
  const videoConstraints = {
    width: 720,
    height: 720,
    facingMode: "user"
  };

  // Simulate face detection with a timer
  useEffect(() => {
    if (isCameraReady && !isScanning) {
      const timer = setTimeout(() => {
        setFaceDetected(true);
      }, 1500);
      
      return () => clearTimeout(timer);
    }
  }, [isCameraReady, isScanning]);

  // Show face scan animation
  useEffect(() => {
    if (isCameraReady && !isScanning) {
      setShowFaceScan(true);
      const timer = setTimeout(() => {
        setShowFaceScan(false);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [isCameraReady, isScanning]);

  const cropToSquare = (imageSrc: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        // Draw the image at 224x224
        ctx.drawImage(img, 0, 0, 224, 224);

        // Convert to base64
        resolve(canvas.toDataURL('image/jpeg', 0.9));
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = imageSrc;
    });
  };

  const capture = useCallback(async () => {
    if (!webcamRef.current) {
      const err = 'Webcam not initialized';
      console.error(err);
      setError(err);
      return;
    }

    try {
      // Set the webcam screenshot size to match our desired output
      if (webcamRef.current.video) {
        webcamRef.current.video.width = 224;
        webcamRef.current.video.height = 224;
      }

      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        // Process the image to ensure it's exactly 224x224
        const processedImage = await cropToSquare(imageSrc);
        
        try {
          const base64Data = processedImage.split(',')[1];
          atob(base64Data);
          
          // Create a test Blob to verify data
          const byteString = atob(base64Data);
          const mimeString = processedImage.split(':')[1].split(';')[0];
          
          const ab = new ArrayBuffer(byteString.length);
          const ia = new Uint8Array(ab);
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
          }
          const blob = new Blob([ab], { type: mimeString });
          console.log('Captured image size:', blob.size, 'bytes');
        } catch (e) {
          console.error('Invalid base64 data:', e);
          setError('Invalid image data format');
          return;
        }
        
        onCapture(processedImage);
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
    setIsCameraReady(true);
    setError(null);

    // Set initial video size
    if (webcamRef.current?.video) {
      webcamRef.current.video.width = 224;
      webcamRef.current.video.height = 224;
    }
  }, []);

  const handleUserMediaError = useCallback((err: string | DOMException) => {
    const errorMessage = err instanceof DOMException ? err.message : err;
    console.error('Camera error:', errorMessage);
    setError(`Camera error: ${errorMessage}`);
    setIsCameraReady(false);
  }, []);

  // Container style with square aspect ratio
  const containerStyle = {
    width: '100%',
    maxWidth: '400px',
    margin: '0 auto',
    aspectRatio: '1/1',
    backgroundColor: isDarkMode ? '#1f1f1f' : '#f0f2f5',
    borderRadius: '12px',
    overflow: 'hidden',
    position: 'relative' as const,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    border: `1px solid ${isDarkMode ? '#333' : '#e0e0e0'}`,
  };

  return (
    <div style={containerStyle}>
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
      
      {/* Face detection overlay */}
      {isCameraReady && !isScanning && showFaceScan && (
        <div style={{
          position: 'absolute',
          top: '0',
          left: '0',
          right: '0',
          bottom: '0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
          zIndex: 2,
        }}>
          {/* Scanning animation */}
          <div style={{
            width: '100%',
            height: '100%',
            border: '2px solid #1DA1F2',
            animation: 'pulse 2s infinite',
            position: 'relative',
            borderRadius: '8px',
          }}>
            {/* Scanning line */}
            <div style={{
              position: 'absolute',
              top: '0',
              left: '0',
              right: '0',
              height: '2px',
              background: 'linear-gradient(90deg, transparent, #1DA1F2, transparent)',
              animation: 'scanLine 2s linear infinite',
              opacity: 0.7,
            }} />
          </div>
        </div>
      )}
      
      {/* Face detected indicator */}
      {faceDetected && !isScanning && (
        <div style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          background: 'rgba(82, 196, 26, 0.8)',
          color: 'white',
          padding: '4px 12px',
          borderRadius: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          fontSize: '14px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
          zIndex: 3,
        }}>
          <CheckOutlined /> Face Detected
        </div>
      )}
      
      {/* Loading overlay */}
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
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(4px)',
          zIndex: 4,
        }}>
          <Spin indicator={<LoadingOutlined style={{ fontSize: 48, color: '#1DA1F2' }} spin />} />
        </div>
      )}

      {/* Camera initialization state */}
      {!isCameraReady && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          padding: '20px',
          background: isDarkMode ? 'rgba(0, 0, 0, 0.7)' : 'rgba(255, 255, 255, 0.9)',
          borderRadius: '12px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          zIndex: 5,
        }}>
          <CameraOutlined style={{ fontSize: 48, color: '#8c8c8c' }} />
          <Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
            {error || 'Initializing camera...'}
          </Text>
        </div>
      )}

      {/* Capture button */}
      {isCameraReady && !isScanning && (
        <div style={{
          position: 'absolute',
          bottom: '16px',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          zIndex: 6,
        }}>
          <Button
            type="primary"
            icon={<CameraOutlined />}
            onClick={capture}
            size="large"
            style={{
              height: '64px',
              width: '64px',
              borderRadius: '32px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
              border: '4px solid white',
            }}
          />
          {error && (
            <Text type="danger" style={{ 
              display: 'block', 
              marginTop: '12px', 
              textAlign: 'center',
              background: isDarkMode ? 'rgba(0, 0, 0, 0.7)' : 'rgba(255, 255, 255, 0.9)',
              padding: '4px 12px',
              borderRadius: '4px',
            }}>
              {error}
            </Text>
          )}
        </div>
      )}

      {/* CSS animations */}
      <style>
        {`
        @keyframes pulse {
          0% {
            transform: scale(0.98);
            box-shadow: 0 0 0 0 rgba(29, 161, 242, 0.7);
          }
          
          70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(29, 161, 242, 0);
          }
          
          100% {
            transform: scale(0.98);
            box-shadow: 0 0 0 0 rgba(29, 161, 242, 0);
          }
        }
        
        @keyframes scanLine {
          0% {
            top: 0%;
          }
          50% {
            top: 100%;
          }
          100% {
            top: 0%;
          }
        }
        `}
      </style>
    </div>
  );
} 