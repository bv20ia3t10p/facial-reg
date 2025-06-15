import { Form, Input, Button } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';

interface LoginFormProps {
  onSubmit: (values: { username: string; password: string }) => void;
  loading?: boolean;
}

export function LoginForm({ onSubmit, loading }: LoginFormProps) {
  const [form] = Form.useForm();

  return (
    <Form
      form={form}
      onFinish={onSubmit}
      layout="vertical"
      style={{ width: '100%' }}
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
          style={{
            width: '100%',
            height: '40px',
          }}
        >
          Log In
        </Button>
      </Form.Item>
    </Form>
  );
} 