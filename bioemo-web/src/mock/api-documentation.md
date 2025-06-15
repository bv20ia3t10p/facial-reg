# BioEmo API Documentation

This document outlines all the mock API endpoints and their expected JSON responses for the BioEmo web application.

## Authentication Endpoints

### 1. QR Code Authentication
```http
POST /api/auth/qr-code
Content-Type: application/json

{
  "qrCode": "string",
  "timestamp": "2024-03-20T10:30:00Z"
}
```

**Response (Success)**
```json
{
  "success": true,
  "confidence": 0.92,
  "timestamp": "2024-03-20T10:30:05Z",
  "emotion": "neutral",
  "message": "Successfully authenticated",
  "sessionToken": "jwt-token-here"
}
```

**Response (Failure)**
```json
{
  "success": false,
  "confidence": 0.45,
  "timestamp": "2024-03-20T10:30:05Z",
  "message": "Authentication failed - low confidence"
}
```

### 2. Identity Verification
```http
POST /api/auth/verify
Content-Type: application/json

{
  "sessionId": "string",
  "timestamp": "2024-03-20T10:30:00Z"
}
```

**Response**
```json
{
  "success": true,
  "confidence": 0.95,
  "timestamp": "2024-03-20T10:30:07Z",
  "emotion": "happy",
  "message": "Identity verified successfully"
}
```

## User Data Endpoints

### 1. Get Current User
```http
GET /api/users/current
Authorization: Bearer <token>
```

**Response**
```json
{
  "id": "user123",
  "name": "John Doe",
  "department": "Engineering",
  "lastAuthenticated": "2024-03-20T09:45:00Z",
  "emotionHistory": [
    {
      "timestamp": "2024-03-20T09:45:00Z",
      "emotion": "neutral",
      "confidence": 0.92
    },
    {
      "timestamp": "2024-03-20T08:30:00Z",
      "emotion": "happy",
      "confidence": 0.85
    },
    {
      "timestamp": "2024-03-20T07:15:00Z",
      "emotion": "tired",
      "confidence": 0.78
    }
  ]
}
```

### 2. Get User Settings
```http
GET /api/users/settings
Authorization: Bearer <token>
```

**Response**
```json
{
  "notifications": {
    "emailAlerts": true,
    "pushNotifications": false
  },
  "privacy": {
    "shareEmotionData": true,
    "anonymizeReports": false
  },
  "authentication": {
    "requireSecondaryVerification": true,
    "sessionTimeout": 3600
  }
}
```

## Dashboard & Analytics Endpoints

### 1. Get Dashboard Stats
```http
GET /api/dashboard/stats
Authorization: Bearer <token>
```

**Response**
```json
{
  "totalAuthentications": 157,
  "averageConfidence": 0.89,
  "emotionBreakdown": {
    "neutral": 45,
    "happy": 35,
    "tired": 25,
    "stressed": 15
  },
  "recentActivity": [
    {
      "success": true,
      "confidence": 0.95,
      "timestamp": "2024-03-20T10:15:00Z",
      "emotion": "neutral",
      "message": "Successfully authenticated"
    },
    {
      "success": true,
      "confidence": 0.88,
      "timestamp": "2024-03-20T09:30:00Z",
      "emotion": "happy",
      "message": "Successfully authenticated"
    },
    {
      "success": false,
      "confidence": 0.45,
      "timestamp": "2024-03-20T08:45:00Z",
      "message": "Authentication failed - low confidence"
    }
  ]
}
```

### 2. Get Emotion Trends
```http
GET /api/analytics/emotion-trends
Authorization: Bearer <token>
Query Parameters:
  - startDate: "2024-03-13"
  - endDate: "2024-03-20"
```

**Response**
```json
{
  "dailyBreakdown": [
    {
      "date": "2024-03-20",
      "emotions": {
        "neutral": 15,
        "happy": 12,
        "tired": 8,
        "stressed": 5
      },
      "averageConfidence": 0.88
    },
    {
      "date": "2024-03-19",
      "emotions": {
        "neutral": 18,
        "happy": 14,
        "tired": 6,
        "stressed": 4
      },
      "averageConfidence": 0.91
    }
  ],
  "trends": {
    "dominantEmotion": "neutral",
    "emotionShifts": [
      {
        "from": "neutral",
        "to": "tired",
        "count": 12,
        "timeOfDay": "afternoon"
      }
    ],
    "peakStressTimes": [
      {
        "dayOfWeek": "Monday",
        "timeOfDay": "morning",
        "stressLevel": 0.65
      }
    ]
  }
}
```

### 3. Get Authentication History
```http
GET /api/analytics/auth-history
Authorization: Bearer <token>
Query Parameters:
  - page: 1
  - limit: 10
  - startDate: "2024-03-13"
  - endDate: "2024-03-20"
```

**Response**
```json
{
  "total": 157,
  "page": 1,
  "limit": 10,
  "data": [
    {
      "id": "auth123",
      "timestamp": "2024-03-20T10:15:00Z",
      "success": true,
      "confidence": 0.95,
      "emotion": "neutral",
      "location": "Main Entrance",
      "device": "Mobile-QR",
      "duration": 2.5
    }
  ],
  "summary": {
    "successRate": 0.92,
    "averageConfidence": 0.89,
    "averageDuration": 2.8
  }
}
```

## Department Analytics Endpoints

### 1. Get Department Overview
```http
GET /api/analytics/department/{departmentId}
Authorization: Bearer <token>
```

**Response**
```json
{
  "departmentId": "eng123",
  "name": "Engineering",
  "metrics": {
    "totalEmployees": 45,
    "activeToday": 38,
    "emotionalState": {
      "overall": "positive",
      "breakdown": {
        "neutral": 40,
        "happy": 35,
        "tired": 15,
        "stressed": 10
      }
    },
    "authenticationStats": {
      "successRate": 0.95,
      "averageConfidence": 0.88,
      "failureReasons": {
        "lowConfidence": 8,
        "systemError": 2,
        "other": 1
      }
    }
  },
  "trends": {
    "weeklyMood": [
      {
        "day": "Monday",
        "dominantEmotion": "neutral",
        "stressLevel": 0.3
      },
      {
        "day": "Tuesday",
        "dominantEmotion": "happy",
        "stressLevel": 0.2
      }
    ],
    "peakTimes": {
      "authentication": {
        "busiest": "09:00-10:00",
        "quietest": "15:00-16:00"
      },
      "stress": {
        "highest": "Monday 09:00-11:00",
        "lowest": "Friday 15:00-17:00"
      }
    }
  }
}
```

## Error Responses

All endpoints may return the following error responses:

### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Invalid or expired token",
  "code": "AUTH001"
}
```

### 403 Forbidden
```json
{
  "error": "forbidden",
  "message": "Insufficient permissions to access this resource",
  "code": "AUTH002"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "Requested resource not found",
  "code": "REQ001"
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "code": "SRV001"
}
``` 