# Phone-Based Biometric Authentication System

A lightweight, privacy-preserving biometric authentication system with federated learning capabilities, optimized for memory efficiency and containerized deployment.

## 🌟 Features

- **Lightweight FastAPI Backend**: Memory-optimized API with CPU-only PyTorch support
- **SQLite Database**: Minimal overhead with built-in admin interface
- **Containerized Architecture**: Docker Compose with service monitoring
- **Memory Optimization**: Designed for systems with limited RAM
- **D: Drive Storage**: All data stored on D: drive for Windows efficiency
- **Real-time Monitoring**: Prometheus + Grafana dashboards
- **Security**: Rate limiting, input validation, and secure headers

## 🏗️ Architecture

```mermaid
graph TB
    Client[Client Applications] --> Nginx[Nginx Load Balancer]
    Nginx --> API[FastAPI Biometric Service]
    API --> SQLite[(SQLite Database)]
    API --> Redis[(Redis Cache)]
    API --> Models[PyTorch Models]
    
    Prometheus[Prometheus] --> API
    Grafana[Grafana] --> Prometheus
    Admin[SQLite Admin] --> SQLite
    
    subgraph "D: Drive Storage"
        Data[/data]
        Cache[/cache]
        Models
        Logs[/logs]
    end
```

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11 (optimized for Windows)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space on D: drive
- **CPU**: 2+ cores recommended

### Software Requirements
- Docker Desktop for Windows
- PowerShell 5.1 or later
- curl (for testing)

## 🚀 Quick Start

### 1. Setup System

Run the setup script as Administrator:

```powershell
# Run as Administrator
.\scripts\setup-system.ps1
```

This will:
- Create directory structure on D: drive
- Copy existing models and data
- Build Docker images
- Configure memory optimization
- Create startup/shutdown scripts

### 2. Start Services

```batch
.\start-system.bat
```

### 3. Test API

```batch
.\test-api.bat
```

## 📚 API Endpoints

### Health Check
```http
GET /health
```

### User Enrollment
```http
POST /enroll
Content-Type: multipart/form-data

user_id: string
image: file (JPEG/PNG, max 5MB)
```

### Authentication
```http
POST /authenticate
Content-Type: multipart/form-data

user_id: string
image: file (JPEG/PNG, max 5MB)
```

### User History
```http
GET /users/{user_id}/history?limit=50
```

### Metrics (Monitoring)
```http
GET /metrics
```

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| **biometric-api** | 8000 | Main FastAPI application |
| **nginx** | 80 | Load balancer and reverse proxy |
| **redis** | 6379 | Caching and session storage |
| **sqlite-web** | 8080 | Database administration interface |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Monitoring dashboards |

## 📊 Access Points

After starting the system:

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs  
- **Database Admin**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3000 (admin/biometric2024)
- **Prometheus**: http://localhost:9090

## 💾 Memory Optimization

The system is optimized for low memory usage:

- **PyTorch**: CPU-only with 512MB memory chunking
- **Thread Limiting**: 2 threads for CPU operations
- **Redis**: 200MB memory limit with LRU eviction
- **Docker**: Memory limits on all containers
- **Model Caching**: Intelligent model loading with cleanup

### Memory Usage by Service
- API: ~1-2GB (includes PyTorch model)
- Nginx: ~64MB
- Redis: ~200MB  
- SQLite Web: ~64MB
- Prometheus: ~256-512MB
- Grafana: ~256-512MB

**Total System Memory**: ~2.5-4GB

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Memory Optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2

# Paths (D: Drive)
DATA_PATH=D:/data
MODELS_PATH=D:/models
CACHE_PATH=D:/cache
LOGS_PATH=D:/logs

# Database
DB_PATH=D:/data/biometric.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
```

### Docker Compose Override
For systems with <8GB RAM, a `docker-compose.override.yml` is created with stricter memory limits.

## 📁 Directory Structure

```
D:/
├── data/
│   ├── biometric.db          # SQLite database
│   └── partitioned/          # Training data (if available)
├── models/
│   └── best_pretrained_model.pth  # PyTorch model
├── cache/                    # Temporary processing files
├── logs/
│   ├── api.log              # Application logs
│   └── nginx/               # Nginx logs
├── redis-data/              # Redis persistence
├── prometheus-data/         # Metrics storage
└── grafana-data/           # Dashboard configs
```

## 🛠️ Development

### Local Development
1. Install Python 3.11+
2. Install dependencies: `pip install -r requirements.txt`
3. Run API: `python api/main.py`

### Adding New Endpoints
1. Edit `api/main.py`
2. Rebuild container: `docker-compose build biometric-api`
3. Restart: `docker-compose restart biometric-api`

### Custom Model Integration
1. Place model in `D:/models/`
2. Update model path in `BiometricProcessor.model_path`
3. Implement model-specific inference in `_process_image_sync()`

## 📈 Monitoring

### Key Metrics
- API response times
- Memory usage per service
- Database query performance
- Authentication success rates
- Error rates and types

### Grafana Dashboards
Access at http://localhost:3000 with admin/biometric2024:
- System Overview
- API Performance
- Memory Usage
- Authentication Analytics

## 🔒 Security Features

- **Rate Limiting**: 10 req/s general, 5 req/s auth endpoints
- **Input Validation**: File type and size limits
- **Security Headers**: XSS, CSRF, and frame protection
- **Network Isolation**: Docker network segmentation
- **Access Control**: Metrics endpoints restricted to internal IPs

## 🧪 Testing

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Enroll user (requires image file)
curl -X POST "http://localhost:8000/enroll" \
  -F "user_id=test_user" \
  -F "image=@path/to/image.jpg"

# Authenticate user
curl -X POST "http://localhost:8000/authenticate" \
  -F "user_id=test_user" \
  -F "image=@path/to/image.jpg"
```

### Load Testing
```bash
# Install Apache Bench
# Test health endpoint
ab -n 100 -c 10 http://localhost:8000/health
```

## 🐛 Troubleshooting

### Common Issues

**1. Docker Build Fails**
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

**2. Out of Memory**
```bash
# Check memory usage
docker stats
# Restart with override
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

**3. Port Already in Use**
```bash
# Check port usage
netstat -ano | findstr :8000
# Kill process or change port in docker-compose.yml
```

**4. Model Not Found**
- Ensure `best_pretrained_model.pth` is in `D:/models/`
- Check file permissions
- Verify file is not corrupted

### Logs
```bash
# View API logs
docker-compose logs biometric-api

# View all service logs
docker-compose logs

# Real-time logs
docker-compose logs -f
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Create an issue with system info and error logs

---

**Note**: This system is optimized for Windows with D: drive storage. For other platforms, adjust paths in configuration files accordingly. 