# BioEmo: Federated Biometric Authentication & Emotion Analysis

[![Architecture](https://mermaid.ink/svg/pako:eNqNVMtqwzAQ_JdCchxSQttd9tJDhw6BAinUqVqSYlFkyTUkAxf-vbOLSdImhV6yePfuzt5xQk6ZIBQ0QhS_CAlwZgA77A1_jYwNq5Z1A7xN-Q-9gU3QWvW32S_8xUv_i_J8SjJzQ0u-4J_wDccG-hUq5UeJ6bYQ2uEwIu1vFq1S3yB3zJzC6j7UDR43y5kC-v_4q_Uq8_bV0Y7vQJtIu_mKckLhNlO6WJp8xW7N97H6dOqD5L-1yM9WjP1cI_jU-Jk6IeJ8x7uG-Fqf9zP60B8Jb-Ew4b3hH-b5L9qj_N6-J7h-s6xV-z91sP6F4J3d1hBf-I9W2zT_w5wG0W0R8sQ4g3GjK2hXm6-A4G2S1q4E1T2j6D-p7gq2gR-X1RjC6oG8e2uQzK6R5r92e0uU7F9-7X7d0WwE_eC7cI)](https://mermaid.live/edit#pako:eNqNVMtqwzAQ_JdCchxSQttd9tJDhw6BAinUqVqSYlFkyTUkAxf-vbOLSdImhV6yePfuzt5xQk6ZIBQ0QhS_CAlwZgA77A1_jYwNq5Z1A7xN-Q-9gU3QWvW32S_8xUv_i_J8SjJzQ0u-4J_wDccG-hUq5UeJ6bYQ2uEwIu1vFq1S3yB3zJzC6j7UDR43y5kC-v_4q_Uq8_bV0Y7vQJtIu_mKckLhNlO6WJp8xW7N97H6dOqD5L-1yM9WjP1cI_jU-Jk6IeJ8x7uG-Fqf9zP60B8Jb-Ew4b3hH-b5L9qj_N6-J7h-s6xV-z91sP6F4J3d1hBf-I9W2zT_w5wG0W0R8sQ4g3GjK2hXm6-A4G2S1q4E1T2j6D-p7gq2gR-X1RjC6oG8e2uQzK6R5r92e0uU7F9-7X7d0WwE_eC7cI)

BioEmo is a comprehensive system for **privacy-preserving biometric authentication** using facial recognition, combined with **emotion analysis** to gauge user well-being. It is built on a foundation of federated learning to ensure user data remains on client devices, protecting privacy while continuously improving the central model.

The system is composed of a modern web interface for interaction, and a set of backend microservices that handle authentication, emotion detection, and the federated learning process.

## Architecture

The project uses a microservices architecture orchestrated with Docker Compose.

-   **Web Interface (`bioemo-web`)**: A React/TypeScript frontend that serves as the user's entry point to the system. It communicates with the backend APIs for authentication and analysis.
-   **Biometric Service (`client-api`)**: A FastAPI service that handles user registration, facial recognition, and participates as a client in the federated learning network. Multiple instances of this service can be run to simulate different data silos (e.g., `client1`, `client2`).
-   **Federated Coordinator (`fl-coordinator`)**: The central server that orchestrates the federated learning process. It aggregates model updates from the various `client-api` instances without ever accessing their raw data.
-   **Emotion Recognition Service (`emotion-api`)**: A dedicated, GPU-accelerated service for analyzing facial emotions from images.

## Key Features

-   **Federated Learning**: Model training is decentralized. User data never leaves the client, and only model updates are sent to a central server, ensuring a high degree of privacy.
-   **Biometric Authentication**: Secure facial recognition for user authentication.
-   **Emotion Detection**: Real-time analysis of facial expressions to monitor user well-being.
-   **Modern Web Interface**: A clean, responsive dashboard for interacting with the system, viewing analytics, and managing users.
-   **Microservices Architecture**: A scalable and maintainable backend built with Docker and FastAPI.

## Tech Stack

| Component            | Technology                                                              |
| -------------------- | ----------------------------------------------------------------------- |
| **Backend**          | Python, FastAPI, PyTorch, SQLAlchemy                                    |
| **Frontend**         | React, TypeScript, Vite, Mantine UI, Axios                              |
| **Privacy Tech**     | Federated Learning, Differential Privacy                                |
| **Infrastructure**   | Docker, Docker Compose, NVIDIA GPU (required)                           |

## Prerequisites

-   **Docker & Docker Compose**: For running the backend microservices. [Install Docker](https://docs.docker.com/get-docker/).
-   **NVIDIA GPU & NVIDIA Container Toolkit**: Required for the machine learning models in the `client-api` and `emotion-api` services. [Install Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
-   **Node.js & npm**: For running the `bioemo-web` frontend in a development environment. [Install Node.js](https://nodejs.org/en/download/).

## How to Run the Project

The application is designed to be run with Docker Compose for the backend and a local development server for the frontend.

### 1. Configure the Frontend

The web interface needs to know where the backend APIs are located.

```bash
# Navigate to the web app directory
cd bioemo-web

# Copy the example environment file
cp env.example .env
```

The default values in `.env` are configured to work with the `docker-compose.yml` setup.

-   `VITE_AUTH_SERVER_URL=http://localhost:8001` (points to `client2-api`)
-   `VITE_EMOTION_SERVER_URL=http://localhost:1236` (points to `emotion-api`)

### 2. Launch Backend Services

From the project's root directory, start all backend services using Docker Compose.

```bash
# This will build the images and start the containers.
# The --build flag is only necessary the first time or after code changes.
docker-compose up --build -d
```

This command starts:
-   `fl-coordinator` on port `9000`
-   `client1-api` on port `8080`
-   `client2-api` on port `8001` (this is the one the UI talks to by default)
-   `emotion-api` on port `1236`

You can view logs for all services with `docker-compose logs -f`.

### 3. Run the Web Frontend

In a new terminal, navigate to the `bioemo-web` directory to start the frontend development server.

```bash
cd bioemo-web

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 4. Access the Application

-   **Web Application**: [http://localhost:5173](http://localhost:5173)
-   **Biometric API (Client 2)**: [http://localhost:8001/docs](http://localhost:8001/docs)
-   **Emotion API**: [http://localhost:1236/docs](http://localhost:1236/docs)
-   **Federated Coordinator**: [http://localhost:9000/docs](http://localhost:9000/docs)

## Project Structure

```
facial-reg/
├── api/                # Source code for the main Python backend (clients and coordinator)
├── bioemo-web/         # React/TypeScript frontend application
├── emo-api/            # Standalone Python service for emotion recognition
├── scripts/            # Helper scripts for data management and setup
├── docker-compose.yml  # Defines and orchestrates all backend services
├── Dockerfile.client   # Dockerfile for the client biometric services
└── Dockerfile.coordinator # Dockerfile for the federated learning coordinator
```

## Configuration

### Backend Services

The configuration for the backend services (ports, environment variables, etc.) is managed entirely within the `docker-compose.yml` file. Key variables include database URLs, JWT secrets, and federated learning parameters.

### Frontend Service

The frontend is configured via the `.env` file inside the `bioemo-web` directory. These variables are prefixed with `VITE_` and are loaded by the Vite development server.

## Scripts and Data Setup

Before running the application, you may need to generate databases, mappings, and ensure the correct model files are in place.

### Required File Structure

For the Docker containers to build and run successfully, certain files and directories must exist locally. Below is the required structure within your project root:

```
facial-reg/
├── data/
│   ├── identity_mapping.json      # Generated by script
│   └── partitioned/
│       ├── client1/
│       │   └── ... (user image folders)
│       ├── client2/
│       │   └── ... (user image folders)
│       └── server/
│           └── ... (user image folders)
├── database/
│   ├── client1.db                 # Generated by script
│   └── client2.db                 # Generated by script
├── emo-api/
│   └── ferplus_ema.pth            # MUST be downloaded/placed here
└── models/
    ├── best_pretrained_model.pth    # Required for all services
    ├── client1_model.pth          # Required for client1
    ├── client2_model.pth          # Required for client2
    └── server_model.pth           # Required for coordinator
```

-   **`data/partitioned/`**: This directory holds the datasets for training the models, split by client. Each subdirectory (`client1`, `client2`, `server`) should contain folders named by user ID, with their corresponding images inside.
-   **`models/`**: This directory must contain the pre-trained models required by the services.
-   **`emo-api/ferplus_ema.pth`**: The emotion recognition model must be placed inside the `emo-api` directory.

### Utility Scripts

The scripts in the `/scripts` directory are used to prepare the data and databases for the application.

1.  **`generate_federated_mapping.py`**
    -   **Purpose**: Scans the `data/partitioned/` directory to create a master `identity_mapping.json` file. This file ensures that each user identity is mapped to a unique integer, which is critical for the federated learning model.
    -   **When to run**: Run this script anytime the user identities in the `data/partitioned/` directories change.
    -   **Usage**: `python scripts/generate_federated_mapping.py`

2.  **`create_client_dbs.py`**
    -   **Purpose**: Populates the client-specific SQLite databases (`client1.db`, `client2.db`) with realistic fake data. It generates users and a history of authentication logs based on the user IDs found in `data/partitioned/`.
    -   **When to run**: Run this script after setting up your user images in the `data/partitioned/` directory and generating the mapping file. This prepares the databases that the `client-api` services will use.
    -   **Usage**: `python scripts/create_client_dbs.py`

The other scripts (`analyze_data_distribution.py` and `fix_class_mapping.py`) are analysis and utility tools that are not required for the initial setup. 