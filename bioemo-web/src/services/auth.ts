// Token management utilities

// Store token in localStorage
export const setAuthToken = (token: string) => {
  localStorage.setItem('authToken', token);
};

// Get token from localStorage
export const getAuthToken = (): string | null => {
  return localStorage.getItem('authToken');
};

// Remove token from localStorage
export const removeAuthToken = () => {
  localStorage.removeItem('authToken');
};

// Check if user is authenticated
export const isAuthenticated = (): boolean => {
  return !!getAuthToken();
};

// Handle logout process
export const logout = (): void => {
  removeAuthToken();
  // We'll clear the user context in the component that calls this function
};

export const getCurrentUserId = (): string => {
  const token = getAuthToken();
  if (!token) return '';
  
  try {
    // Check if it's a mock token (format: mock-token-userId)
    if (token.startsWith('mock-token-')) {
      return token.replace('mock-token-', '');
    }
    
    // Otherwise try to decode as JWT token
    const base64Url = token.split('.')[1];
    if (!base64Url) {
      console.warn('Invalid token format');
      return '';
    }
    
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
      return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

    return JSON.parse(jsonPayload).sub || '';
  } catch (e) {
    console.error('Error decoding token:', e);
    return '';
  }
}; 