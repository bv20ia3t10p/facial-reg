import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { Row, Col, Space, Spin, Alert, Button } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { getUserInfo } from '../services/api';
import type { User, UserInfo, EmotionPrediction } from '../types';
import { toast } from 'react-hot-toast';
import { UserProfile as UserProfileComponent } from '../components/UserProfile';

interface LocationState {
  user: User;
  userInfo: UserInfo;
  capturedImage: string;
  emotions: EmotionPrediction;
  authResult: any;
  lastAuthenticated: string;
}

export const UserProfile: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as LocationState | undefined;
  
  const [userInfo, setUserInfo] = useState<UserInfo | null>(state?.userInfo || null);
  const [isLoading, setIsLoading] = useState(!state?.userInfo);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserData = async () => {
      if (!userId) {
        setError('No user ID provided');
        setIsLoading(false);
        return;
      }

      // If we already have userInfo from state, use it
      if (state?.userInfo) {
        setUserInfo(state.userInfo);
        setIsLoading(false);
        return;
      }

      try {
        const info = await getUserInfo(userId);
        setUserInfo(info);
      } catch (err) {
        console.error('Failed to fetch user info:', err);
        setError('Failed to load user information');
        toast.error('Failed to load user information');
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, [userId, state]);

  const handleBack = () => {
    navigate('/');
  };

  if (!userId) {
    return (
      <Alert
        message="Error"
        description="No user ID provided"
        type="error"
        showIcon
      />
    );
  }

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error || !userInfo) {
    return (
      <Alert
        message="Error"
        description={error || 'Failed to load user information'}
        type="error"
        showIcon
      />
    );
  }

  return (
    <Row justify="center" style={{ padding: '24px' }}>
      <Col xs={24} lg={20} xl={18}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Button 
            icon={<ArrowLeftOutlined />} 
            onClick={handleBack}
            style={{ marginBottom: '16px' }}
          >
            Back to Home
          </Button>
          
          <UserProfileComponent
            user={state?.user || {
              id: userInfo.user_id,
              name: userInfo.name,
              email: userInfo.email,
              department: userInfo.department,
              role: userInfo.role,
              joinDate: userInfo.enrolled_at,
              lastAuthenticated: userInfo.last_authenticated
            }}
            userInfo={userInfo}
            emotions={state?.emotions || userInfo.emotional_state}
            isLoading={isLoading}
          />
        </Space>
      </Col>
    </Row>
  );
};

export default UserProfile; 