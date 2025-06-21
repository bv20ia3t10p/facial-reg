import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { Layout } from 'antd';
import { ThemeProvider } from './contexts/ThemeContext';
import { UserProvider } from './contexts/UserContext';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import AppRoutes from './routes';
import { useTheme } from './contexts/ThemeContext';

const { Content } = Layout;

const AppContent: React.FC = () => {
  const { isDarkMode } = useTheme();
  
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header />
      <Content style={{
        marginTop: 64,
        padding: '24px',
        background: isDarkMode ? '#192734' : '#F7F9FA',
      }}>
        <AppRoutes />
      </Content>
      <Footer />
    </Layout>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <UserProvider>
      <Router>
        <AppContent />
      </Router>
      </UserProvider>
    </ThemeProvider>
  );
};

export default App;
