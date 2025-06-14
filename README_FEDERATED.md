# Federated Learning Biometric Authentication System

A privacy-preserving biometric authentication system implementing **Federated Learning (FL)**, **Homomorphic Encryption (HE)**, and **Differential Privacy (DP)** in a hybrid architecture with 1 server and 2 clients.

## 🏗️ Architecture Overview

### Hybrid Architecture Components

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Biometric     │    │   Federated          │    │   Shared        │
│   API System    │◄──►│   Learning Service   │◄──►│   Storage       │
│                 │    │                      │    │                 │
│ • Authentication│    │ • FL Coordinator     │    │ • Global Models │
│ • User Enrollment│   │ • Client Management  │    │ • Training Data │
│ • Model Serving │    │ • Secure Aggregation │    │ • Logs & Metrics│
└─────────────────┘    └──────────────────────┘    └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐         ┌───────▼────────┐
            │   FL Client 1  │         │   FL Client 2  │
            │                │         │                │
            │ • Local Training│        │ • Local Training│
            │ • HE Encryption │        │ • HE Encryption │
            │ • DP Protection │        │ • DP Protection │
            └────────────────┘         └────────────────┘
```

### Key Features

- **🔐 Privacy-Preserving**: Implements FL + HE + DP for maximum privacy
- **🏢 Hybrid Architecture**: Separates authentication API from federated training
- **💾 Memory Optimized**: Designed for systems with limited RAM (4-8GB)
- **📊 Production Ready**: Includes monitoring, logging, and health checks
- **🐳 Containerized**: Full Docker deployment with orchestration
- **🔄 Automated**: Background model updates and federated rounds

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** with PowerShell
- **Docker Desktop** installed and running
- **8GB+ RAM** recommended (4GB minimum)
- **10GB+ free disk space** on D: drive

### 1. Setup System

```powershell
# Clone and navigate to repository
cd D:\Repo\facial-reg

# Run automated setup (creates directories, builds images, starts services)
.\scripts\setup-federated-system.ps1 -StartServices

# Or setup without starting services
.\scripts\setup-federated-system.ps1
```

### 2. Start Services

```powershell
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Verify System

```powershell
# Run comprehensive tests
.\scripts\test-federated-system.ps1

# Test with verbose output
.\scripts\test-federated-system.ps1 -Verbose

# Trigger a federated learning round
.\scripts\test-federated-system.ps1 -TriggerRound
```

## 🔧 System Components

### Core Services

| Service | Port | Description |
|---------|------|-------------|
| **Biometric API** | 8000 | Main authentication API with federated integration |
| **Federated Coordinator** | 8001 | FL coordinator managing training rounds |
| **Federated Client 1** | - | FL client with local training data |
| **Federated Client 2** | - | FL client with local training data |
| **Nginx** | 80/443 | Reverse proxy with rate limiting |
| **Redis** | 6379 | Caching and session storage |
| **SQLite Web** | 8080 | Database administration interface |
| **Prometheus** | 9090 | Metrics collection and monitoring |
| **Grafana** | 3000 | Monitoring dashboards |

### Access URLs

- **API Documentation**: http://localhost:8000/docs
- **Federated Status**: http://localhost:8000/federated/status
- **Database Admin**: http://localhost:8080
- **Monitoring Dashboard**: http://localhost:3000 (admin/admin123)
- **Metrics**: http://localhost:9090

## 🔒 Privacy & Security Features

### Federated Learning (FL)
- **Decentralized Training**: Models trained locally on client devices
- **No Raw Data Sharing**: Only encrypted model updates are shared
- **Secure Aggregation**: Server aggregates encrypted updates
- **Client Selection**: Dynamic client participation based on availability

### Homomorphic Encryption (HE)
- **CKKS Scheme**: Supports encrypted arithmetic operations
- **Encrypted Model Updates**: Client updates encrypted before transmission
- **Server-Side Aggregation**: Aggregation performed on encrypted data
- **Key Management**: Secure key distribution and management

### Differential Privacy (DP)
- **Noise Addition**: Gaussian noise added to gradients
- **Privacy Budget**: ε-differential privacy with budget tracking
- **Per-Client Budgets**: Individual privacy budgets for each client
- **Adaptive Noise**: Noise scales with sensitivity and privacy requirements

## 📊 API Endpoints

### Authentication Endpoints

```http
POST /enroll
Content-Type: multipart/form-data
{
  "user_id": "string",
  "image": "file"
}

POST /authenticate
Content-Type: multipart/form-data
{
  "user_id": "string", 
  "image": "file"
}

GET /users/{user_id}/history
```

### Federated Learning Endpoints

```http
GET /federated/status
# Returns comprehensive FL system status

POST /federated/trigger-round
# Manually triggers a federated learning round

GET /federated/model-history
# Returns history of model updates

POST /admin/reload-model
# Forces API to reload the biometric model
```

### Coordinator Endpoints

```http
GET /health
# Coordinator health check

POST /clients/register
# Register new FL client

GET /clients/active
# List active FL clients

POST /rounds/start
# Start new federated round

GET /rounds/current
# Get current round status

POST /rounds/{round_id}/submit
# Submit encrypted model update

GET /models/global
# Get global model information

GET /privacy/status/{client_id}
# Get privacy budget status
```

## 🔄 Federated Learning Workflow

### 1. Client Registration
```mermaid
sequenceDiagram
    participant C as FL Client
    participant S as FL Coordinator
    
    C->>S: Register with capabilities
    S->>C: Return HE public context
    S->>S: Store client metadata
```

### 2. Federated Round
```mermaid
sequenceDiagram
    participant API as Biometric API
    participant S as FL Coordinator  
    participant C1 as Client 1
    participant C2 as Client 2
    
    API->>S: Trigger round (if conditions met)
    S->>S: Start new round
    S->>C1: Round notification
    S->>C2: Round notification
    
    C1->>C1: Local training with DP
    C1->>C1: Encrypt model updates (HE)
    C1->>S: Submit encrypted updates
    
    C2->>C2: Local training with DP
    C2->>C2: Encrypt model updates (HE)
    C2->>S: Submit encrypted updates
    
    S->>S: Aggregate encrypted updates
    S->>S: Update global model
    S->>API: Notify model update
    API->>API: Reload model
```

### 3. Privacy Protection
```mermaid
graph TD
    A[Raw Training Data] --> B[Local Training]
    B --> C[Gradient Computation]
    C --> D[Add DP Noise]
    D --> E[Clip Gradients]
    E --> F[HE Encryption]
    F --> G[Encrypted Updates]
    G --> H[Secure Aggregation]
    H --> I[Global Model Update]
```

## 📁 Directory Structure

```
D:/
├── data/                    # All persistent data
│   ├── client1/            # Client 1 training data
│   │   ├── identity_1/     # Identity folders
│   │   ├── identity_2/
│   │   └── config.json     # Client configuration
│   ├── client2/            # Client 2 training data
│   │   ├── identity_6/
│   │   ├── identity_7/
│   │   └── config.json
│   ├── federated/          # Federated learning data
│   ├── redis/              # Redis persistence
│   ├── prometheus/         # Metrics storage
│   └── grafana/            # Dashboard data
├── models/                 # Model storage
│   ├── federated_model.pth # Current federated model
│   ├── biometric_model.pth # API model
│   └── backup/             # Model backups
└── logs/                   # Application logs
    ├── api/                # API logs
    ├── federated/          # Coordinator logs
    └── clients/            # Client logs
```

## 🛠️ Configuration

### Environment Variables

```bash
# API Configuration
PYTHONPATH=/app
TORCH_HOME=/app/cache/torch

# Memory Limits (Docker)
API_MEMORY=1g
COORDINATOR_MEMORY=800m
CLIENT_MEMORY=600m

# Privacy Parameters
MAX_PRIVACY_BUDGET=50.0
NOISE_MULTIPLIER=1.0
MAX_GRAD_NORM=1.0

# FL Parameters
MIN_CLIENTS=2
MAX_CLIENTS=10
ROUND_TIMEOUT=300
TARGET_ACCURACY=0.6
```

### Client Configuration

```json
{
  "client_id": "client1",
  "client_type": "mobile",
  "data_path": "/app/data/client1",
  "privacy_budget": 50.0,
  "local_epochs": 1,
  "batch_size": 8
}
```

## 📈 Monitoring & Logging

### Grafana Dashboards
- **System Overview**: Memory, CPU, network usage
- **FL Metrics**: Round progress, client participation
- **Privacy Metrics**: Budget consumption, noise levels
- **API Metrics**: Authentication rates, success rates

### Log Files
```powershell
# View real-time logs
docker-compose logs -f federated-coordinator
docker-compose logs -f federated-client1
docker-compose logs -f federated-client2
docker-compose logs -f biometric-api

# Log locations
D:/logs/api/api.log
D:/logs/federated/federated.log
D:/logs/clients/client1.log
D:/logs/clients/client2.log
```

## 🧪 Testing & Validation

### Automated Tests
```powershell
# Full system test
.\scripts\test-federated-system.ps1

# Test with round trigger
.\scripts\test-federated-system.ps1 -TriggerRound

# Verbose testing
.\scripts\test-federated-system.ps1 -Verbose
```

### Manual Testing
```powershell
# Test API health
curl http://localhost:8000/health

# Test federated status
curl http://localhost:8000/federated/status

# Trigger federated round
curl -X POST http://localhost:8000/federated/trigger-round

# Check active clients
curl http://localhost:8001/clients/active

# Check privacy status
curl http://localhost:8001/privacy/status/client1
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Services Not Starting
```powershell
# Check Docker status
docker ps -a

# Check logs for errors
docker-compose logs federated-coordinator

# Restart services
docker-compose down
docker-compose up -d
```

#### 2. Memory Issues
```powershell
# Check memory usage
docker stats

# Reduce memory limits in docker-compose.yml
# Restart with lower limits
```

#### 3. Client Registration Issues
```powershell
# Check coordinator logs
docker-compose logs federated-coordinator

# Verify client data directories exist
ls D:/data/client1
ls D:/data/client2

# Check network connectivity
docker network ls
```

#### 4. Privacy Budget Exhausted
```powershell
# Check privacy status
curl http://localhost:8001/privacy/status/client1

# Reset privacy budgets (restart clients)
docker-compose restart federated-client1 federated-client2
```

### Performance Optimization

#### Memory Optimization
- Reduce batch sizes in client configs
- Lower model complexity
- Increase cleanup intervals
- Use CPU-only PyTorch

#### Training Optimization
- Adjust learning rates
- Reduce local epochs
- Optimize data loading
- Use gradient compression

## 📚 Technical Details

### Dependencies
- **PyTorch**: Deep learning framework (CPU-only for memory efficiency)
- **TenSEAL**: Homomorphic encryption library
- **Opacus**: Differential privacy for PyTorch
- **FastAPI**: Modern web framework for APIs
- **SQLAlchemy**: Database ORM
- **Redis**: In-memory caching
- **Docker**: Containerization platform

### Security Considerations
- All model updates encrypted with HE
- Differential privacy protects individual data points
- No raw data leaves client devices
- Secure aggregation prevents inference attacks
- Privacy budgets prevent excessive information leakage

### Scalability
- Horizontal scaling: Add more FL clients
- Vertical scaling: Increase memory/CPU limits
- Load balancing: Nginx handles API requests
- Database scaling: SQLite for simplicity, can migrate to PostgreSQL

## 🤝 Contributing

### Development Setup
```powershell
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server
python -m api.main
python -m federated.coordinator
```

### Adding New Clients
1. Create client data directory: `D:/data/client3/`
2. Add client service to `docker-compose.yml`
3. Update monitoring configuration
4. Test client registration

### Extending Privacy Features
1. Implement new DP mechanisms in `federated/coordinator.py`
2. Add HE schemes in `federated/client.py`
3. Update privacy tracking in database
4. Add monitoring metrics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TenSEAL** for homomorphic encryption capabilities
- **Opacus** for differential privacy implementation
- **FastAPI** for the excellent web framework
- **Docker** for containerization support

---

**Note**: This system is designed for research and development purposes. For production deployment, additional security hardening and compliance measures should be implemented.
 