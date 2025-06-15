import { Modal, Form, Input, Button } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useState } from 'react';
import type { PasswordVerificationProps } from '../types';
import { api } from '../services/api';

export function PasswordVerification({
  visible,
  onClose,
  onSuccess,
}: PasswordVerificationProps) {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (values: { username: string; password: string }) => {
    try {
      setLoading(true);
      const result = await api.verifyCredentials(values.username, values.password);
      if (result.success && result.user) {
        onSuccess(result.user);
      } else {
        form.setFields([
          {
            name: 'password',
            errors: ['Invalid credentials'],
          },
        ]);
      }
    } catch (error) {
      form.setFields([
        {
          name: 'password',
          errors: ['Error verifying credentials'],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      title="Additional Verification Required"
      open={visible}
      onCancel={onClose}
      footer={null}
    >
      <Form
        form={form}
        onFinish={handleSubmit}
        layout="vertical"
      >
        <Form.Item
          name="username"
          rules={[{ required: true, message: 'Please enter your username' }]}
        >
          <Input
            prefix={<UserOutlined />}
            placeholder="Username"
            size="large"
          />
        </Form.Item>

        <Form.Item
          name="password"
          rules={[{ required: true, message: 'Please enter your password' }]}
        >
          <Input.Password
            prefix={<LockOutlined />}
            placeholder="Password"
            size="large"
          />
        </Form.Item>

        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            style={{ width: '100%' }}
          >
            Verify
          </Button>
        </Form.Item>
      </Form>
    </Modal>
  );
} 