import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

// Import pages
import { Home } from './pages/Home';
import Auth from './pages/Auth';
import { Analytics } from './pages/Analytics';
import { AddUser } from './pages/AddUser';
import HRDashboard from './pages/HRDashboard';
import { VerificationRequests } from './pages/VerificationRequests';
import { UserProfile } from './pages/UserProfile';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const AppRoutes: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/authentication" element={<Auth />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/add-user" element={<AddUser />} />
        <Route path="/hr-dashboard" element={<HRDashboard />} />
        <Route path="/verification-requests" element={<VerificationRequests />} />
        <Route path="/profile/:userId" element={<UserProfile />} />
        {/* Settings route will be added later */}
      </Routes>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#333',
            color: '#fff',
          },
          success: {
            iconTheme: {
              primary: '#22c55e',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </QueryClientProvider>
  );
};

export default AppRoutes; 