# Docker Deployment Guide

Complete Docker setup for the ML Recommendation API with PostgreSQL database.

---

## 📦 What's Included

This Docker setup provides:

- **PostgreSQL 14** - Production database with persistent storage
- **ML API** - FastAPI application with all ML models
- **pgAdmin** (optional) - Web-based database management
- **Redis** (optional) - Production rate limiting
- **Automated Migration** - Database setup and data import
- **Health Checks** - Automatic service monitoring
- **Volume Management** - Persistent data storage
- **Network Isolation** - Secure inter-container communication

---

## 📁 Docker Files Overview

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Main Docker Compose configuration |
| `docker-compose.dev.yml` | Development overrides (hot reload) |
| `ml_api/Dockerfile` | ML API container definition |
| `.env.docker.template` | Environment variables template |
| `.dockerignore` | Exclude files from Docker build |
| `DOCKER_SETUP.md` | Complete setup documentation |
| `DOCKER_QUICKSTART.md` | Quick start guide |

---

## 🚀 Quick Start

**See [DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md) for the fastest setup.**

### 1. Setup Environment

```bash
cp .env.docker.template .env
# Edit .env and set required values
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Run Migration

```bash
docker-compose exec ml-api python -m ml_api.scripts.migrate_all
```

### 4. Access API

Open http://localhost:8000/docs

---

## 🎯 Use Cases

### Development

```bash
# Start with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Code changes auto-reload the server
# No need to rebuild after Python file edits
```

### Testing

```bash
# Run tests inside container
docker-compose exec ml-api pytest ml_api/tests/ -v

# Run specific test
docker-compose exec ml-api pytest ml_api/tests/test_db_models.py -v
```

### Production

```bash
# Start with production profiles
docker-compose --profile production up -d

# Includes Redis for rate limiting
# Optimized for performance
```

### Database Management

```bash
# Start with pgAdmin
docker-compose --profile tools up -d

# Access pgAdmin at http://localhost:5050
# Add server connection to 'postgres' service
```

---

## 🔧 Configuration

### Essential Environment Variables

Edit `.env` file:

```env
# Required
POSTGRES_PASSWORD=your_secure_password
JWT_SECRET_KEY=your_secret_key_32_chars_minimum
ADMIN_PASSWORD=your_admin_password

# Optional
API_PORT=8000
POSTGRES_PORT=5432
ALLOWED_ORIGINS=http://localhost:3000
```

### Port Mapping

| Service | Internal Port | External Port (configurable) |
|---------|--------------|------------------------------|
| ML API | 8000 | 8000 (.env: API_PORT) |
| PostgreSQL | 5432 | 5432 (.env: POSTGRES_PORT) |
| pgAdmin | 80 | 5050 (.env: PGADMIN_PORT) |
| Redis | 6379 | 6379 (.env: REDIS_PORT) |

---

## 📊 Services Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Docker Network                         │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │            │    │            │    │            │        │
│  │   ML API   │◄───┤ PostgreSQL │    │   Redis    │        │
│  │  (FastAPI) │    │   (DB)     │    │ (Optional) │        │
│  │            │    │            │    │            │        │
│  └─────┬──────┘    └─────▲──────┘    └────────────┘        │
│        │                 │                                   │
└────────┼─────────────────┼───────────────────────────────────┘
         │                 │
    Port 8000         Port 5432
         │                 │
    ┌────▼────┐       ┌────▼────┐
    │         │       │         │
    │ Browser │       │ pgAdmin │
    │  /API   │       │  (Web)  │
    │ Clients │       │         │
    └─────────┘       └─────────┘
```

---

## 💾 Data Persistence

### Named Volumes

Docker creates named volumes for persistent data:

```bash
# List volumes
docker volume ls | grep aliaa

# Volumes created:
# - postgres_data    (Database files)
# - ml_models        (Trained models)
# - ml_data          (Application data)
# - ml_logs          (Log files)
```

### Backup

```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres -Fc ml_recommendation > backup.dump

# Backup volume
docker run --rm -v aliaa-graduation-project_postgres_data:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/postgres_backup.tar.gz -C /data .
```

### Restore

```bash
# Restore database
docker-compose exec -T postgres pg_restore -U postgres -d ml_recommendation -c < backup.dump

# Restore volume
docker run --rm -v aliaa-graduation-project_postgres_data:/data \
  -v $(pwd):/backup alpine \
  tar xzf /backup/postgres_backup.tar.gz -C /data
```

---

## 🔍 Monitoring

### Health Checks

Docker automatically monitors service health:

```bash
# View health status
docker-compose ps

# Services show:
# - healthy (green) - Service is running correctly
# - unhealthy (red) - Service has issues
# - starting (yellow) - Service is initializing
```

### Logs

```bash
# Real-time logs
docker-compose logs -f

# Specific service
docker-compose logs -f ml-api

# Last 100 lines
docker-compose logs --tail=100 ml-api

# With timestamps
docker-compose logs -t ml-api
```

### Resource Usage

```bash
# Monitor CPU, memory, network
docker stats

# For specific containers
docker stats ml-recommendation-api ml-recommendation-db
```

---

## 🛡️ Security

### Production Security Checklist

- [ ] Use strong passwords (20+ characters)
- [ ] Generate new JWT secret: `openssl rand -hex 32`
- [ ] Don't expose PostgreSQL port in production
- [ ] Enable HTTPS with reverse proxy (nginx/Traefik)
- [ ] Update `ALLOWED_ORIGINS` to production domain
- [ ] Use Docker secrets for sensitive data
- [ ] Enable log rotation
- [ ] Regularly update images: `docker-compose pull`
- [ ] Set up automated backups
- [ ] Monitor for security vulnerabilities

### Network Isolation

Services communicate via Docker network:
- PostgreSQL is NOT exposed to host in production
- Only API port is accessible externally
- Inter-container communication uses service names

---

## 🚨 Troubleshooting

### Common Issues

**1. "Port already in use"**
```bash
# Change port in .env
API_PORT=8001

# Restart
docker-compose down && docker-compose up -d
```

**2. "Cannot connect to database"**
```bash
# Check PostgreSQL is healthy
docker-compose ps postgres

# Wait for startup (can take 30 seconds)
docker-compose exec postgres pg_isready -U postgres
```

**3. "Migration fails"**
```bash
# Ensure CSV files exist
ls ratings_Beauty.csv content/product_descriptions.csv

# Check files are mounted
docker-compose exec ml-api ls /app/ratings_Beauty.csv
```

**4. "Out of memory"**
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory (8GB recommended)

# Reduce pool size in .env
DB_POOL_SIZE=3
```

### Getting Help

```bash
# View all logs
docker-compose logs > full_logs.txt

# Check service health
docker-compose ps

# Inspect service
docker-compose exec ml-api bash
```

---

## 🔄 Updates and Maintenance

### Update Application Code

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose build ml-api
docker-compose up -d ml-api
```

### Update Dependencies

```bash
# Edit ml_api/requirements.txt
# Then rebuild
docker-compose build --no-cache ml-api
docker-compose up -d ml-api
```

### Update PostgreSQL

```bash
# Backup first!
docker-compose exec postgres pg_dump -U postgres -Fc ml_recommendation > backup.dump

# Update image version in docker-compose.yml
# postgres:14-alpine → postgres:15-alpine

# Restart
docker-compose up -d postgres
```

---

## 📚 Documentation Links

- **[DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md)** - 5-minute setup guide
- **[DOCKER_SETUP.md](./DOCKER_SETUP.md)** - Complete documentation
- **[ML_API_DEPLOYMENT.md](./ML_API_DEPLOYMENT.md)** - Full deployment guide
- **[CLAUDE.md](./CLAUDE.md)** - Project overview and API documentation

---

## 💡 Tips

### Development Workflow

```bash
# 1. Make code changes in ml_api/
# 2. Watch logs for auto-reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f ml-api

# 3. Test changes immediately (no rebuild needed!)
curl http://localhost:8000/health
```

### Production Best Practices

```bash
# Use specific image tags (not 'latest')
image: ml-api:1.0.0

# Set resource limits
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G

# Use health checks
healthcheck:
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## 📞 Support

- **Issues:** Check logs with `docker-compose logs`
- **Documentation:** See DOCKER_SETUP.md for detailed help
- **Docker Docs:** https://docs.docker.com/
- **Docker Compose:** https://docs.docker.com/compose/

---

**Created:** January 23, 2026
**Last Updated:** January 23, 2026
**Version:** 1.0.0
