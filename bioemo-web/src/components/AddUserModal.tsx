import { Modal, Form, Input, Select, Button, Upload, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { api } from '../services/api';

const { Option } = Select;

interface AddUserModalProps {
  visible: boolean;
  onClose: () => void;
}

interface UserForm {
  name: string;
  email: string;
  department: string;
  role: string;
  images: UploadFile[];
}

export function AddUserModal({ visible, onClose }: AddUserModalProps) {
  const [form] = Form.useForm<UserForm>();

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      await api.addUser(values);
      message.success('User added successfully');
      form.resetFields();
      onClose();
    } catch (error) {
      message.error('Failed to add user');
    }
  };

  return (
    <Modal
      title="Add New User"
      open={visible}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>
          Cancel
        </Button>,
        <Button key="submit" type="primary" onClick={handleSubmit}>
          Add User
        </Button>,
      ]}
    >
      <Form
        form={form}
        layout="vertical"
        requiredMark="optional"
      >
        <Form.Item
          name="name"
          label="Name"
          rules={[{ required: true, message: 'Please enter the name' }]}
        >
          <Input placeholder="Enter full name" />
        </Form.Item>

        <Form.Item
          name="email"
          label="Email"
          rules={[
            { required: true, message: 'Please enter the email' },
            { type: 'email', message: 'Please enter a valid email' }
          ]}
        >
          <Input placeholder="Enter email address" />
        </Form.Item>

        <Form.Item
          name="department"
          label="Department"
          rules={[{ required: true, message: 'Please select the department' }]}
        >
          <Select placeholder="Select department">
            <Option value="engineering">Engineering</Option>
            <Option value="marketing">Marketing</Option>
            <Option value="sales">Sales</Option>
            <Option value="hr">Human Resources</Option>
            <Option value="management">Management</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="role"
          label="Role"
          rules={[{ required: true, message: 'Please select the role' }]}
        >
          <Select placeholder="Select role">
            <Option value="employee">Employee</Option>
            <Option value="manager">Manager</Option>
            <Option value="admin">Administrator</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="images"
          label="Face Images"
          rules={[{ required: true, message: 'Please upload at least one image' }]}
        >
          <Upload
            listType="picture"
            maxCount={5}
            multiple
            beforeUpload={() => false} // Prevent auto upload
          >
            <Button icon={<UploadOutlined />}>Upload Face Images</Button>
          </Upload>
        </Form.Item>
      </Form>
    </Modal>
  );
} 