import { Table, Button, Space, Modal, Typography, Tag, Image } from 'antd';
import { CheckOutlined, CloseOutlined, EyeOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { api } from '../services/api';
import type { VerificationRequest } from '../types';

const { Text, Title } = Typography;

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

  const handleApprove = async () => {
    if (!selectedRequest) return;
    await api.approveVerificationRequest(selectedRequest.id);
    await onRequestProcessed();
  };

  const handleReject = async () => {
    if (!selectedRequest) return;
    await api.rejectVerificationRequest(selectedRequest.id);
    await onRequestProcessed();
  };

  const columns = [
    {
      title: 'Employee ID',
      dataIndex: 'employeeId',
      key: 'employeeId',
    },
    {
      title: 'Reason',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
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
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'Action',
      key: 'action',
      render: (_: any, record: VerificationRequest) => (
        <Button
          icon={<EyeOutlined />}
          onClick={() => setSelectedRequest(record)}
        >
          View Details
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