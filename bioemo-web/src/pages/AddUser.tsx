import { Card, Typography, Form, Input, Button, Select, Space, message, Upload, Modal } from 'antd';
import { UserAddOutlined, CameraOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import { useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../services/api';
import type { RcFile, UploadFile } from 'antd/es/upload/interface';

const { Title, Text } = Typography;
const { Option } = Select;

interface UserForm {
  name: string;
  email: string;
  department: string;
  images: File[];
}

const departments = [
  'Engineering',
  'Marketing',
  'Sales',
  'Human Resources',
  'Finance',
  'Operations',
  'Research',
  'Customer Support',
];

const getBase64 = (file: RcFile): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });

export function AddUser() {
  const [form] = Form.useForm<UserForm>();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState('');
  const [previewTitle, setPreviewTitle] = useState('');

  const addUserMutation = useMutation({
    mutationFn: (values: UserForm) => api.addUser({
      ...values,
      images: fileList.map(file => file.originFileObj!),
    }),
    onSuccess: () => {
      message.success('User added successfully');
      form.resetFields();
      setFileList([]);
    },
    onError: () => {
      message.error('Failed to add user');
    },
  });

  const handlePreview = async (file: UploadFile) => {
    if (!file.url && !file.preview) {
      file.preview = await getBase64(file.originFileObj as RcFile);
    }

    setPreviewImage(file.url || (file.preview as string));
    setPreviewOpen(true);
    setPreviewTitle(file.name);
  };

  const handleChange = ({ fileList: newFileList }: { fileList: UploadFile[] }) => {
    setFileList(newFileList);
  };

  const uploadButton = (
    <div>
      <PlusOutlined />
      <div style={{ marginTop: 8 }}>Add Photo</div>
    </div>
  );

  const handleSubmit = (values: any) => {
    if (fileList.length === 0) {
      message.error('Please add at least one photo');
      return;
    }
    addUserMutation.mutate(values);
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Title level={2} style={{ margin: 0, fontWeight: 600 }}>Add New User</Title>

      <Card
        bordered={false}
        style={{ borderRadius: '16px', maxWidth: 600 }}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          requiredMark={false}
        >
          <Form.Item
            name="name"
            label="Full Name"
            rules={[{ required: true, message: 'Please enter the user\'s name' }]}
          >
            <Input 
              size="large" 
              placeholder="Enter full name"
              style={{ borderRadius: 8 }}
            />
          </Form.Item>

          <Form.Item
            name="email"
            label="Email Address"
            rules={[
              { required: true, message: 'Please enter the user\'s email' },
              { type: 'email', message: 'Please enter a valid email' }
            ]}
          >
            <Input 
              size="large" 
              placeholder="Enter email address"
              style={{ borderRadius: 8 }}
            />
          </Form.Item>

          <Form.Item
            name="department"
            label="Department"
            rules={[{ required: true, message: 'Please select a department' }]}
          >
            <Select
              size="large"
              placeholder="Select department"
              style={{ borderRadius: 8 }}
            >
              {departments.map(dept => (
                <Option key={dept} value={dept}>{dept}</Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            label={
              <Space>
                <Text>Face Photos</Text>
                <Text type="secondary">(Add multiple photos for better recognition)</Text>
              </Space>
            }
          >
            <Upload
              listType="picture-card"
              fileList={fileList}
              onPreview={handlePreview}
              onChange={handleChange}
              beforeUpload={() => false}
              accept="image/*"
              multiple
            >
              {fileList.length >= 8 ? null : uploadButton}
            </Upload>
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, marginTop: 24 }}>
            <Button
              type="primary"
              htmlType="submit"
              icon={<UserAddOutlined />}
              size="large"
              loading={addUserMutation.isPending}
              style={{
                width: '100%',
                height: 48,
                borderRadius: 24,
                fontSize: 16,
              }}
            >
              Add User
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Modal
        open={previewOpen}
        title={previewTitle}
        footer={null}
        onCancel={() => setPreviewOpen(false)}
      >
        <img alt="preview" style={{ width: '100%' }} src={previewImage} />
      </Modal>
    </Space>
  );
} 