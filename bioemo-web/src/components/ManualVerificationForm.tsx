import { Modal, Form, Input, Button, Space, Alert } from 'antd';
import { api } from '../services/api';
import { useState } from 'react';

const { TextArea } = Input;

interface ManualVerificationFormProps {
  visible: boolean;
  onClose: () => void;
  capturedImage?: string;
  confidence: number;
}

interface VerificationRequest {
  employeeId: string;
  reason: string;
  additionalNotes?: string;
  capturedImage: string;
  confidence: number;
}

export function ManualVerificationForm({ visible, onClose, capturedImage, confidence }: ManualVerificationFormProps) {
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    try {
      setSubmitting(true);
      const values = await form.validateFields();
      
      const request: VerificationRequest = {
        ...values,
        capturedImage: capturedImage || '',
        confidence,
      };

      await api.submitManualVerification(request);
      form.resetFields();
      onClose();
    } catch (error) {
      // Form validation error is handled by antd
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="Request Manual Verification"
      open={visible}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>
          Cancel
        </Button>,
        <Button 
          key="submit" 
          type="primary" 
          loading={submitting}
          onClick={handleSubmit}
        >
          Submit Request
        </Button>,
      ]}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Alert
          message="Low Confidence Authentication"
          description="Your verification attempt had low confidence. Please submit this form for manual verification by HR."
          type="info"
          showIcon
        />

        <Form
          form={form}
          layout="vertical"
          requiredMark="optional"
        >
          <Form.Item
            name="employeeId"
            label="Employee ID"
            rules={[{ required: true, message: 'Please enter your employee ID' }]}
          >
            <Input placeholder="Enter your employee ID" />
          </Form.Item>

          <Form.Item
            name="reason"
            label="Reason for Manual Verification"
            rules={[{ required: true, message: 'Please select a reason' }]}
          >
            <Input placeholder="e.g., System didn't recognize me, Appearance change, etc." />
          </Form.Item>

          <Form.Item
            name="additionalNotes"
            label="Additional Notes"
          >
            <TextArea 
              rows={4}
              placeholder="Any additional information that might help with verification"
            />
          </Form.Item>

          {capturedImage && (
            <Form.Item label="Captured Image">
              <img 
                src={capturedImage} 
                alt="Captured face"
                style={{ 
                  width: '100%', 
                  maxWidth: '300px', 
                  borderRadius: '8px',
                  border: '1px solid #d9d9d9'
                }} 
              />
            </Form.Item>
          )}
        </Form>
      </Space>
    </Modal>
  );
}
