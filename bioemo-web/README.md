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

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

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
