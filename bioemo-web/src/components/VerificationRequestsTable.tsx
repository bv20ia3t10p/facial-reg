import { Table, Button, Space, Modal, Typography, Tag, Image, theme, Input, Form, message, Alert } from 'antd';
import { CheckOutlined, CloseOutlined, EyeOutlined, KeyOutlined, CopyOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { updateVerificationRequestStatus } from '../services/api';
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

function OTPModal({ visible, onClose, otp }: { visible: boolean; onClose: () => void; otp: string }) {
  return (
    <Modal
      title="OTP Generated"
      open={visible}
      onCancel={onClose}
      footer={[
        <Button key="close" type="primary" onClick={onClose}>
          Close
        </Button>
      ]}
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Alert
          message="Verification Approved"
          description="Please provide this OTP to the user for authentication."
          type="success"
          showIcon
        />
        
        <div style={{ textAlign: 'center', padding: '20px 0' }}>
          <Title level={2}>{otp}</Title>
          <Text type="secondary">This OTP will be valid for 10 minutes</Text>
        </div>
        
        <Form.Item label="OTP">
          <Input.Group compact>
            <Input
              style={{ width: 'calc(100% - 40px)' }}
              value={otp}
              readOnly
            />
            <Button
              icon={<CopyOutlined />}
              onClick={() => {
                navigator.clipboard.writeText(otp);
                message.success('OTP copied to clipboard');
              }}
            />
          </Input.Group>
        </Form.Item>
      </Space>
    </Modal>
  );
}

function RequestDetailsModal({ request, visible, onClose, onApprove, onReject }: RequestDetailsModalProps) {
  const [loading, setLoading] = useState(false);
  const [showOtp, setShowOtp] = useState(false);
  const [otp, setOtp] = useState("");

  const handleAction = async (action: 'approve' | 'reject') => {
    setLoading(true);
    try {
      if (action === 'approve') {
        // Generate OTP
        const generatedOtp = Math.floor(100000 + Math.random() * 900000).toString();
        setOtp(generatedOtp);
        
        await onApprove();
        setShowOtp(true);
      } else {
        await onReject();
        onClose();
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
    <Modal
      title="Verification Request Details"
        open={visible && !showOtp}
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
            Approve & Generate OTP
        </Button>,
      ]}
      width={800}
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div>
          <Title level={5}>Employee</Title>
          <Text>{request.user?.name || 'N/A'}</Text>
        </div>
        
        <div>
          <Title level={5}>Department</Title>
          <Text>{request.user?.department || 'N/A'}</Text>
        </div>

        <div>
          <Title level={5}>Submitted At</Title>
          <Text>{new Date(request.submittedAt).toLocaleString()}</Text>
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
            src={request.capturedImage && !request.capturedImage.startsWith('data:') 
              ? `data:image/jpeg;base64,${request.capturedImage}` 
              : request.capturedImage}
            alt="Captured face"
            style={{ maxWidth: '400px', borderRadius: '8px' }}
          />
        </div>
      </Space>
    </Modal>

    <OTPModal 
      visible={showOtp} 
      onClose={() => {
        setShowOtp(false);
        onClose();
      }} 
      otp={otp}
    />
    </>
  );
}

export function VerificationRequestsTable({ requests, loading, onRequestProcessed }: VerificationRequestsTableProps) {
  const [selectedRequest, setSelectedRequest] = useState<VerificationRequest | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const { token } = useToken();

  const handleViewRequest = (record: VerificationRequest) => {
    setSelectedRequest(record);
    setModalVisible(true);
  };

  const handleApproveRequest = async () => {
    if (!selectedRequest) return;

    try {
      // Call API to approve the request
      await updateVerificationRequestStatus(selectedRequest.id, 'approved');
      await onRequestProcessed();
    } catch (error) {
      console.error("Error approving request:", error);
      message.error("Failed to approve verification request");
    }
  };

  const handleRejectRequest = async () => {
    if (!selectedRequest) return;

    try {
      // Call API to reject the request
      await updateVerificationRequestStatus(selectedRequest.id, 'rejected');
      await onRequestProcessed();
    } catch (error) {
      console.error("Error rejecting request:", error);
      message.error("Failed to reject verification request");
    }
  };

  const columns = [
    {
      title: 'Employee',
      dataIndex: ['user', 'name'],
      key: 'employeeName',
    },
    {
      title: 'Department',
      dataIndex: ['user', 'department'],
      key: 'department',
    },
    {
      title: 'Reason',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
    },
    {
      title: 'Time',
      dataIndex: 'submittedAt',
      key: 'submittedAt',
      render: (timestamp: string) => new Date(timestamp).toLocaleString(),
    },
    {
      title: 'Processed At',
      dataIndex: 'processedAt',
      key: 'processedAt',
      render: (timestamp: string | null) => timestamp ? new Date(timestamp).toLocaleString() : 'N/A',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        let color = 'default';
        if (status === 'pending') {
          color = 'orange';
        } else if (status === 'approved') {
          color = 'green';
        } else if (status === 'rejected') {
          color = 'red';
        }
        return <Tag color={color}>{status.charAt(0).toUpperCase() + status.slice(1)}</Tag>;
      },
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => {
        const percent = (confidence * 100).toFixed(1);
        let color = token.colorError;
        if (confidence >= 0.8) {
          color = token.colorSuccess;
        } else if (confidence >= 0.6) {
          color = token.colorWarning;
        }
        return (
          <Text style={{ color }}>
            {percent}%
          </Text>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: VerificationRequest) => (
        <Space size="small">
          <Button 
            type="primary" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => handleViewRequest(record)}
          >
            View
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <>
      <Table 
        dataSource={requests} 
        columns={columns} 
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 10 }}
      />
      {selectedRequest && (
        <RequestDetailsModal
          request={selectedRequest}
          visible={modalVisible}
          onClose={() => setModalVisible(false)}
          onApprove={handleApproveRequest}
          onReject={handleRejectRequest}
        />
      )}
    </>
  );
}