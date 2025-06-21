import React, { createContext, useContext, useState, useEffect } from 'react';
import type { UserInfo } from '../types';
import { getCurrentUserId } from '../services/auth';
import { getUserInfo } from '../services/api';

interface UserContextType {
  currentUser: UserInfo | null;
  isHRDepartment: boolean;
  isLoading: boolean;
  error: string | null;
  refreshUserData: () => Promise<void>;
  clearUserData: () => void;
}

const UserContext = createContext<UserContextType>({
  currentUser: null,
  isHRDepartment: false,
  isLoading: true,
  error: null,
  refreshUserData: async () => {},
  clearUserData: () => {}
});

export const useUser = () => useContext(UserContext);

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Determine if user is in HR department - check both department and role fields
  const isHRDepartment = Boolean(
    currentUser && (
      (currentUser.department?.toLowerCase().includes('hr') || 
       currentUser.department?.toLowerCase().includes('human resources')) ||
      (currentUser.role?.toLowerCase().includes('hr') || 
       currentUser.role?.toLowerCase().includes('human resources'))
    )
  );

  const refreshUserData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const userId = getCurrentUserId();
      if (!userId) {
        setCurrentUser(null);
        setIsLoading(false);
        return;
      }
      
      const userInfo = await getUserInfo(userId);
      setCurrentUser(userInfo);
      console.log("Fetched user info:", userInfo);
      console.log("Is HR department:", Boolean(
        userInfo && (
          (userInfo.department?.toLowerCase().includes('hr') || 
           userInfo.department?.toLowerCase().includes('human resources')) ||
          (userInfo.role?.toLowerCase().includes('hr') || 
           userInfo.role?.toLowerCase().includes('human resources'))
        )
      ));
      
    } catch (err) {
      console.error('Failed to fetch user info:', err);
      setError('Failed to fetch user info');
      setCurrentUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const clearUserData = () => {
    setCurrentUser(null);
  };
  
  // Load user data on initial mount
  useEffect(() => {
    refreshUserData();
  }, []);

  return (
    <UserContext.Provider 
      value={{ 
        currentUser, 
        isHRDepartment, 
        isLoading, 
        error,
        refreshUserData,
        clearUserData
      }}
    >
      {children}
    </UserContext.Provider>
  );
}; 