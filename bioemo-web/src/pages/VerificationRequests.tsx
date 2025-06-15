import { Typography, Space } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { VerificationRequestsTable } from '../components/VerificationRequestsTable';

const { Title } = Typography;

export function VerificationRequests() {
  const { data: requests, isLoading, refetch } = useQuery({
    queryKey: ['verificationRequests'],
    queryFn: () => api.getVerificationRequests(),
  });

  const pendingRequests = requests?.filter(req => req.status === 'pending') || [];

  const handleRequestProcessed = async () => {
    await refetch();
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Title level={2}>Verification Requests</Title>
      <VerificationRequestsTable
        requests={pendingRequests}
        loading={isLoading}
        onRequestProcessed={handleRequestProcessed}
      />
    </Space>
  );
} 