import { Modal, Form, Input, Select, Button, Upload, message, Spin, Steps, Divider, Switch, Space, Alert, Typography } from 'antd';
import { UploadOutlined, UserOutlined, MailOutlined, TeamOutlined, LockOutlined, FileImageOutlined, PlusOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { useState } from 'react';
import { api } from '../services/api';
import { useTheme } from '../contexts/ThemeContext';

const { Option } = Select;
const { Step } = Steps;

interface AddUserModalProps {
  visible: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

interface UserForm {
  name: string;
  email: string;
  department: string;
  role: string;
  password: string;
  confirmPassword: string;
  images: UploadFile[];
}

export function AddUserModal({ visible, onClose, onSuccess }: AddUserModalProps) {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const { isDarkMode } = useTheme();
  const [imageCount, setImageCount] = useState(0);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      
      setLoading(true);
      
      // Check if at least one image is uploaded
      if (!values.images || values.images.length === 0) {
        message.error('Please upload at least one face image');
        setLoading(false);
        return;
      }
      
      // Create form data for multipart/form-data request
      const formData = new FormData();
      formData.append('name', values.name);
      formData.append('email', values.email);
      formData.append('department', values.department);
      formData.append('role', values.role);
      
      // Append all images
      values.images.forEach((file: any) => {
        formData.append('images', file.originFileObj);
      });
      
      // Submit the form
      const response = await api.registerUser(formData);
      
      if (response.success) {
        message.success('User registered successfully with default password: demo');
        form.resetFields();
        if (onSuccess) {
          onSuccess();
        }
        onClose();
      } else {
        message.error(response.message || 'Failed to register user');
      }
    } catch (error) {
      console.error('Registration error:', error);
      message.error('Failed to register user');
    } finally {
      setLoading(false);
    }
  };

  const nextStep = async () => {
    try {
      // Validate all fields and submit
      await form.validateFields();
      handleSubmit();
    } catch (error) {
      console.error('Validation error:', error);
    }
  };

  const handleImagesChange = (info: any) => {
    setImageCount(info.fileList.length);
    form.setFieldsValue({ images: info.fileList });
  };

  const renderStepContent = () => {
    return (
      <>
        <Form.Item
          name="name"
          label="Full Name"
          rules={[{ required: true, message: 'Please enter the name' }]}
        >
          <Input 
            prefix={<UserOutlined style={{ color: '#1DA1F2' }} />} 
            placeholder="Enter full name" 
            size="large"
          />
        </Form.Item>

        <Form.Item
          name="email"
          label="Email Address"
          rules={[
            { required: true, message: 'Please enter the email' },
            { type: 'email', message: 'Please enter a valid email' }
          ]}
        >
          <Input 
            prefix={<MailOutlined style={{ color: '#1DA1F2' }} />} 
            placeholder="Enter email address" 
            size="large"
          />
        </Form.Item>

        <Form.Item
          name="department"
          label="Department"
          rules={[{ required: true, message: 'Please select the department' }]}
        >
          <Select 
            placeholder="Select department" 
            size="large"
            suffixIcon={<TeamOutlined style={{ color: '#1DA1F2' }} />}
          >
            <Option value="Engineering">Engineering</Option>
            <Option value="Marketing">Marketing</Option>
            <Option value="Sales">Sales</Option>
            <Option value="HR">Human Resources</Option>
            <Option value="Management">Management</Option>
            <Option value="Finance">Finance</Option>
            <Option value="IT">IT</Option>
            <Option value="Operations">Operations</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="role"
          label="Role"
          rules={[{ required: true, message: 'Please select the role' }]}
        >
          <Select 
            placeholder="Select role" 
            size="large"
          >
            <Option value="employee">Employee</Option>
            <Option value="manager">Manager</Option>
            <Option value="admin">Administrator</Option>
          </Select>
        </Form.Item>
        
        <Alert
          message="Default Password"
          description="All new users will be created with the default password: 'demo'"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          name="images"
          label="Face Images"
          valuePropName="fileList"
          getValueFromEvent={(e) => e && e.fileList}
          rules={[{ required: true, message: 'Please upload at least one face image' }]}
        >
          <Upload
            listType="picture-card"
            beforeUpload={() => false}
            onChange={handleImagesChange}
            multiple
            accept="image/*"
          >
            <div>
              <PlusOutlined />
              <div style={{ marginTop: 8 }}>Upload</div>
            </div>
          </Upload>
        </Form.Item>
        
        <div style={{ marginTop: 16 }}>
          <Typography.Text type="secondary">
            Upload clear face images for biometric authentication. 
            We recommend at least 3 different angles.
          </Typography.Text>
        </div>
      </>
    );
  };

  const getFooterButtons = () => {
    const buttons = [];
    
    if (loading) {
      return [<Spin key="loading" />];
    }
    
    if (currentStep < 2) {
      buttons.push(
        <Button key="next" type="primary" onClick={nextStep} disabled={loading}>
          Register User
        </Button>
      );
    } else {
      buttons.push(
        <Button key="submit" type="primary" onClick={handleSubmit} loading={loading}>
          Register User
        </Button>
      );
    }
    
    if (currentStep > 0) {
      buttons.unshift(
        <Button key="back" onClick={() => setCurrentStep(currentStep - 1)} disabled={loading}>
          Back
        </Button>
      );
    }
    
    return buttons;
  };

  return (
    <Modal
      title={
        <div style={{ textAlign: 'center', padding: '8px 0' }}>
          <h2 style={{ margin: 0 }}>Register New User</h2>
          <p style={{ margin: '4px 0 0', opacity: 0.7, fontSize: '14px' }}>
            Add a new user to the facial recognition system
          </p>
        </div>
      }
      open={visible}
      onCancel={onClose}
      width={600}
      footer={[
        <Button key="cancel" onClick={onClose} disabled={loading}>
          Cancel
        </Button>,
        ...getFooterButtons()
      ]}
      bodyStyle={{ padding: '24px' }}
    >
      {!currentStep && (
        <Steps
          current={currentStep}
          items={[
            { title: 'Basic Info', description: 'User details' },
            { title: 'Face Images', description: 'Upload photos' },
          ]}
          style={{ marginBottom: '24px' }}
        />
      )}
      
      <Spin spinning={loading}>
        <Form
          form={form}
          layout="vertical"
          requiredMark="optional"
          style={{ maxHeight: '400px', overflowY: 'auto', padding: '8px' }}
        >
          {renderStepContent()}
        </Form>
      </Spin>
    </Modal>
  );
} 