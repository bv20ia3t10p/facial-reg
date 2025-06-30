import React from 'react';
import { Result, Button } from 'antd';
import { useNavigate } from 'react-router-dom';
import { HomeOutlined, ArrowLeftOutlined } from '@ant-design/icons';

export const NotFound: React.FC = () => {
  const navigate = useNavigate();

  const handleGoHome = () => {
    navigate('/');
  };

  const handleGoBack = () => {
    navigate(-1);
  };

  return (
    <Result
      status="404"
      title="404"
      subTitle="Sorry, the page you visited does not exist."
      extra={
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
          <Button type="primary" icon={<HomeOutlined />} onClick={handleGoHome}>
            Go Home
          </Button>
          <Button icon={<ArrowLeftOutlined />} onClick={handleGoBack}>
            Go Back
          </Button>
        </div>
      }
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}
    />
  );
};