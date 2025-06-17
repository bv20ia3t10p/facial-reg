import { Table, Button, Space, Modal, Typography, Tag, Image, theme } from 'antd';
import { CheckOutlined, CloseOutlined, EyeOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { api } from '../services/api';
import type { VerificationRequest } from '../types';

const { Text, Title } = Typography;
const { useToken } = theme;

interface RequestDetailsModalProps {
  request: VerificationRequest;
  visible: boolean;
  onClose: () => void;
  onApprove: () => Promise<void>;
  onReject: () => Promise<void>;
}

interface VerificationRequestsTableProps {
  requests: VerificationRequest[];
  loading: boolean;
  onRequestProcessed: () => Promise<unknown>;
}

function RequestDetailsModal({ request, visible, onClose, onApprove, onReject }: RequestDetailsModalProps) {
  const [loading, setLoading] = useState(false);

  const handleAction = async (action: 'approve' | 'reject') => {
    setLoading(true);
    try {
      if (action === 'approve') {
        await onApprove();
      } else {
        await onReject();
      }
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      title="Verification Request Details"
      open={visible}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>
          Close
        </Button>,
        <Button 
          key="reject" 
          danger 
          icon={<CloseOutlined />}
          onClick={() => handleAction('reject')}
          loading={loading}
        >
          Reject
        </Button>,
        <Button
          key="approve"
          type="primary"
          icon={<CheckOutlined />}
          onClick={() => handleAction('approve')}
          loading={loading}
        >
          Approve
        </Button>,
      ]}
      width={800}
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div>
          <Title level={5}>Employee ID</Title>
          <Text>{request.employeeId}</Text>
        </div>

        <div>
          <Title level={5}>Reason</Title>
          <Text>{request.reason}</Text>
        </div>

        {request.additionalNotes && (
          <div>
            <Title level={5}>Additional Notes</Title>
            <Text>{request.additionalNotes}</Text>
          </div>
        )}

        <div>
          <Title level={5}>Confidence Score</Title>
          <Text>{(request.confidence * 100).toFixed(1)}%</Text>
        </div>

        <div>
          <Title level={5}>Captured Image</Title>
          <Image
            src={request.capturedImage}
            alt="Captured face"
            style={{ maxWidth: '400px', borderRadius: '8px' }}
          />
        </div>

        <div>
          <Title level={5}>Submitted At</Title>
          <Text>{new Date(request.submittedAt).toLocaleString()}</Text>
        </div>
      </Space>
    </Modal>
  );
}

export function VerificationRequestsTable({ requests, loading, onRequestProcessed }: VerificationRequestsTableProps) {
  const [selectedRequest, setSelectedRequest] = useState<VerificationRequest | null>(null);
  const { token } = useToken();

  const handleApprove = async () => {
    if (!selectedRequest) return;
    await api.updateVerificationRequestStatus(selectedRequest.id, 'approved');
    await onRequestProcessed();
  };

  const handleReject = async () => {
    if (!selectedRequest) return;
    await api.updateVerificationRequestStatus(selectedRequest.id, 'rejected');
    await onRequestProcessed();
  };

  const columns = [
    {
      title: 'Employee ID',
      dataIndex: 'employeeId',
      key: 'employeeId',
      width: 120,
    },
    {
      title: 'Reason',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
      width: '30%',
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const statusConfig = {
          pending: { color: 'gold', text: 'Pending' },
          approved: { color: 'success', text: 'Approved' },
          rejected: { color: 'error', text: 'Rejected' },
        };
        const config = statusConfig[status as keyof typeof statusConfig];
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    {
      title: 'Submitted',
      dataIndex: 'submittedAt',
      key: 'submittedAt',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'Action',
      key: 'action',
      width: 120,
      fixed: 'right' as const,
      render: (_: any, record: VerificationRequest) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => setSelectedRequest(record)}
        >
          View
        </Button>
      ),
    },
  ];

  return (
    <>
      <Table
        columns={columns}
        dataSource={requests}
        loading={loading}
        rowKey="id"
        style={{ 
          marginTop: token.marginMD,
          backgroundColor: token.colorBgContainer,
          borderRadius: token.borderRadiusLG,
          boxShadow: token.boxShadowTertiary
        }}
        pagination={{
          pageSize: 10,
          position: ['bottomCenter'],
          showSizeChanger: true,
          showTotal: (total: number) => `Total ${total} items`
        }}
        scroll={{ x: 'max-content' }}
        onRow={(record) => ({
          onClick: () => setSelectedRequest(record),
          style: {
            cursor: 'pointer',
            transition: 'background-color 0.3s',
            '&:hover': {
              backgroundColor: token.colorFillAlter
            }
          }
        })}
      />

      {selectedRequest && (
        <RequestDetailsModal
          request={selectedRequest}
          visible={true}
          onClose={() => setSelectedRequest(null)}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      )}
    </>
  );
} 