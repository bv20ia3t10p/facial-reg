# BioEmo Web Interface

A modern, Twitter-inspired web interface for the BioEmo facial recognition and emotion detection system. This interface provides a user-friendly way to interact with the biometric authentication system while monitoring employee wellbeing through emotion detection.

## Features

- **Dashboard**: Overview of authentication statistics and emotion trends
- **QR Code Authentication**: Seamless authentication process using QR codes
- **Real-time Emotion Detection**: Monitor and track emotional states during authentication
- **Analytics Dashboard**: Detailed insights into authentication patterns and emotional trends

## Tech Stack

- React 18 with TypeScript
- Vite for fast development and building
- Mantine UI components
- React Router for navigation
- Axios for API communication

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment variables:
   ```bash
   # Copy the example file and edit as needed
   cp env.example .env
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Build for production:
   ```bash
   npm run build
   ```

## Environment Variables

The application uses the following environment variables:

- `VITE_AUTH_SERVER_URL`: Authentication API server URL (default: http://localhost:8001)
- `VITE_EMOTION_SERVER_URL`: Emotion analysis API server URL (default: http://localhost:1235)
- `VITE_USE_MOCK_API`: Whether to use mock data instead of real API calls (default: false)
- `VITE_CONFIDENCE_THRESHOLD`: Confidence threshold for facial recognition (0.0-1.0, default: 0.7)
- `VITE_LOW_CONFIDENCE_THRESHOLD`: Low confidence threshold for verification requests (0.0-1.0, default: 0.3)

Copy `env.example` to `.env` and modify the values as needed for your environment.

## Project Structure

```
bioemo-web/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/         # Page components
│   ├── types/         # TypeScript type definitions
│   ├── mock/          # Mock data for development
│   ├── App.tsx        # Main application component
│   └── main.tsx       # Application entry point
├── public/            # Static assets
└── index.html         # HTML template
```

## Development

The project is set up with TypeScript and follows modern React best practices. Key features include:

- **Type Safety**: Comprehensive TypeScript types for all components and data
- **Component Architecture**: Modular components for easy maintenance
- **Responsive Design**: Mobile-first approach using Mantine UI
- **Mock Data**: Development-ready mock responses for testing

## Future Enhancements

- [ ] Real-time WebSocket connection for live updates
- [ ] Integration with the backend API
- [ ] Advanced emotion analytics and reporting
- [ ] User preference management
- [ ] Admin dashboard for system management
