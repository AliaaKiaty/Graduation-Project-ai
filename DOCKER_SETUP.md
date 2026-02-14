# Docker Setup Guide

Complete guide for deploying the ML Recommendation API with PostgreSQL using Docker.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Database Migration](#database-migration)
6. [Accessing Services](#accessing-services)
7. [Docker Commands Reference](#docker-commands-reference)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## Prerequisites

### Required Software

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
  - Docker version 20.10+ recommended
  - Docker Compose version 2.0+ recommended

**Installation:**
- Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- Linux: [Docker Engine](https://docs.docker.com/engine/install/)

**Verify Installation:**
```bash
docker --version
# Expected: Docker version 20.10.x or higher

docker-compose --version
# Expected: Docker Compose version 2.x.x or higher
```

### System Requirements

- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 10GB available
- **Ports:** 5432 (PostgreSQL), 8000 (API), 5050 (pgAdmin - optional)

---

## Quick Start

### Step 1: Clone and Navigate

```bash
cd "Aliaa Graduation Project"
```

### Step 2: Create Environment File

```bash
# Copy template
cp .env.docker.template .env

# Edit .env with your values
# IMPORTANT: Change these values:
# - POSTGRES_PASSWORD
# - JWT_SECRET_KEY (generate with: openssl rand -hex 32)
# - ADMIN_PASSWORD
```

**Minimum Required Configuration:**
```env
POSTGRES_PASSWORD=your_secure_password_here
JWT_SECRET_KEY=your_secret_key_minimum_32_characters_long
ADMIN_PASSWORD=your_admin_password_here
```

### Step 3: Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

**Expected Output:**
```
[+] Running 3/3
 ✔ Network aliaa-graduation-project_ml-network  Created
 ✔ Container ml-recommendation-db               Started
 ✔ Container ml-recommendation-api              Started
```

### Step 4: Wait for Initialization

```bash
# Check service health
docker-compose ps

# All services should show "healthy" status
# Wait ~30-60 seconds for initial startup
```

### Step 5: Run Database Migration

```bash
# Run migration script inside container
docker-compose exec ml-api python -m ml_api.scripts.migrate_all
```

**Expected Output:**
```
[1/3] Initializing database...
  ✓ Created 6 tables
[2/3] Importing products...
  ✓ Products added: 124,000
[3/3] Importing ratings...
  ✓ Ratings added: 2,000,000
✓ Migration Complete!
```

### Step 6: Verify API is Running

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected: {"status":"healthy", ...}
```

**🎉 Your API is now running!**

---

## Configuration

### Environment Variables

All configuration is done via the `.env` file. See `.env.docker.template` for all available options.

#### Essential Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL password | `my_secure_db_pass123` |
| `JWT_SECRET_KEY` | JWT signing key (32+ chars) | `abc123...` (use `openssl rand -hex 32`) |
| `ADMIN_PASSWORD` | Admin user password | `Admin@2024!Secure` |

#### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `ml_recommendation` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `DB_MAX_OVERFLOW` | `10` | Max overflow connections |

#### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8000` | API port |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS allowed origins |
| `RATE_LIMIT_GENERAL` | `100/minute` | General rate limit |

### Volume Configuration

Docker Compose creates named volumes for persistent data:

| Volume | Purpose | Location |
|--------|---------|----------|
| `postgres_data` | Database data | PostgreSQL data files |
| `ml_models` | ML model files | Saved models (SVD, TF-IDF, etc.) |
| `ml_data` | Application data | SQLite user DB, logs |
| `ml_logs` | Application logs | API access and error logs |

**View volumes:**
```bash
docker volume ls | grep aliaa
```

**Inspect volume:**
```bash
docker volume inspect aliaa-graduation-project_postgres_data
```

---

## Running the Application

### Standard Startup

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f ml-api
```

### Development Mode

```bash
# Start with auto-reload (mount code as volume)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# This will mount your local code directory
# Changes to Python files will reload the server automatically
```

### With Optional Services

```bash
# Start with pgAdmin (database UI)
docker-compose --profile tools up -d

# Start with Redis (production rate limiting)
docker-compose --profile production up -d

# Start with both
docker-compose --profile tools --profile production up -d
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v

# Stop specific service
docker-compose stop ml-api
```

---

## Database Migration

### Initial Migration

After first startup, run the migration:

```bash
docker-compose exec ml-api python -m ml_api.scripts.migrate_all
```

This will:
1. Create database tables
2. Import ~124K products from `content/product_descriptions.csv`
3. Import ~2M ratings from `ratings_Beauty.csv`

**IMPORTANT:** Ensure CSV files exist in project root:
- `ratings_Beauty.csv`
- `content/product_descriptions.csv`

### Individual Migration Steps

```bash
# 1. Initialize database schema only
docker-compose exec ml-api python -m ml_api.scripts.init_database

# 2. Import products only
docker-compose exec ml-api python -m ml_api.scripts.import_products

# 3. Import ratings only
docker-compose exec ml-api python -m ml_api.scripts.import_ratings
```

### Verify Migration

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d ml_recommendation

# Run verification queries
SELECT COUNT(*) FROM products;   -- Should be ~124,000
SELECT COUNT(*) FROM ratings;    -- Should be ~2,000,000
SELECT COUNT(*) FROM categories; -- Should be ~10

# Exit PostgreSQL
\q
```

### Model Retraining

```bash
# Retrain all ML models from database
docker-compose exec ml-api python -m ml_api.scripts.retrain_models
```

**Expected Duration:** 5-15 minutes depending on hardware

---

## Accessing Services

### ML Recommendation API

**Base URL:** `http://localhost:8000`

**Swagger Docs:** `http://localhost:8000/docs`

**ReDoc:** `http://localhost:8000/redoc`

**Health Check:** `http://localhost:8000/health`

### PostgreSQL Database

**Connection Details:**
- Host: `localhost`
- Port: `5432`
- Database: `ml_recommendation`
- User: `postgres`
- Password: (from `.env` file)

**Connect via CLI:**
```bash
docker-compose exec postgres psql -U postgres -d ml_recommendation
```

**Connect via GUI Client:**
- Use any PostgreSQL client (DBeaver, pgAdmin, TablePlus, etc.)
- Connection string: `postgresql://postgres:PASSWORD@localhost:5432/ml_recommendation`

### pgAdmin (Optional - with --profile tools)

**URL:** `http://localhost:5050`

**Credentials:**
- Email: `admin@admin.com` (or from `.env`)
- Password: `admin` (or from `.env`)

**Add Server in pgAdmin:**
1. Right-click "Servers" → "Create" → "Server"
2. General tab:
   - Name: `ML Recommendation DB`
3. Connection tab:
   - Host: `postgres` (Docker network name)
   - Port: `5432`
   - Database: `ml_recommendation`
   - Username: `postgres`
   - Password: (from `.env`)

---

## Docker Commands Reference

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart ml-api

# View service status
docker-compose ps

# View resource usage
docker stats
```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Logs for specific service
docker-compose logs ml-api

# Last 100 lines
docker-compose logs --tail=100 ml-api

# Logs with timestamps
docker-compose logs -t ml-api
```

### Execute Commands in Container

```bash
# Open bash shell in API container
docker-compose exec ml-api bash

# Run Python script
docker-compose exec ml-api python -m ml_api.scripts.migrate_all

# Run pytest
docker-compose exec ml-api pytest ml_api/tests/ -v

# Check Python version
docker-compose exec ml-api python --version
```

### Database Operations

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d ml_recommendation

# Backup database
docker-compose exec postgres pg_dump -U postgres ml_recommendation > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres -d ml_recommendation < backup.sql

# Create database backup (compressed)
docker-compose exec postgres pg_dump -U postgres -Fc ml_recommendation > backup.dump
```

### Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect aliaa-graduation-project_postgres_data

# Remove unused volumes
docker volume prune

# Backup volume
docker run --rm -v aliaa-graduation-project_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volume
docker run --rm -v aliaa-graduation-project_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /data
```

### Image Management

```bash
# Build images
docker-compose build

# Build without cache
docker-compose build --no-cache

# Pull latest images
docker-compose pull

# Remove unused images
docker image prune
```

---

## Troubleshooting

### Issue: Container Fails to Start

**Diagnosis:**
```bash
# Check logs
docker-compose logs ml-api

# Check container status
docker-compose ps
```

**Common Causes:**
1. **Port already in use**
   ```bash
   # Find process using port 8000
   # Windows:
   netstat -ano | findstr :8000

   # Linux/Mac:
   lsof -i :8000

   # Change port in .env
   API_PORT=8001
   ```

2. **Missing environment variables**
   ```bash
   # Check .env file exists
   ls -la .env

   # Verify required variables are set
   cat .env | grep POSTGRES_PASSWORD
   cat .env | grep JWT_SECRET_KEY
   ```

3. **Database connection failed**
   ```bash
   # Check PostgreSQL is running
   docker-compose ps postgres

   # Check PostgreSQL logs
   docker-compose logs postgres

   # Verify DATABASE_URL in .env
   ```

### Issue: Migration Fails

**Error: "CSV file not found"**
```bash
# Ensure CSV files are in correct location
ls ratings_Beauty.csv
ls content/product_descriptions.csv

# Check files are mounted in container
docker-compose exec ml-api ls /app/ratings_Beauty.csv
docker-compose exec ml-api ls /app/content/product_descriptions.csv
```

**Error: "Database connection refused"**
```bash
# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready -U postgres

# Restart services
docker-compose restart
```

**Error: "Permission denied"**
```bash
# Check database permissions
docker-compose exec postgres psql -U postgres -d ml_recommendation -c "GRANT ALL PRIVILEGES ON DATABASE ml_recommendation TO postgres;"
```

### Issue: High Memory Usage

```bash
# Check resource usage
docker stats

# Reduce workers/pool size in .env
DB_POOL_SIZE=3
DB_MAX_OVERFLOW=5

# Restart services
docker-compose restart ml-api
```

### Issue: Slow Response Times

**Diagnosis:**
```bash
# Check database slow queries
docker-compose exec postgres psql -U postgres -d ml_recommendation

# Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 100;
SELECT pg_reload_conf();

# View slow queries
SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;
```

**Solutions:**
```bash
# Add indexes (run inside container)
docker-compose exec ml-api python -m ml_api.scripts.add_indexes

# Increase PostgreSQL shared_buffers
# Edit docker-compose.yml, add to postgres service:
command: postgres -c shared_buffers=256MB -c max_connections=100
```

### Issue: Cannot Access API

**Check network connectivity:**
```bash
# From host machine
curl http://localhost:8000/health

# From another container
docker-compose exec postgres curl http://ml-api:8000/health

# Check Docker network
docker network ls
docker network inspect aliaa-graduation-project_ml-network
```

**Check firewall:**
```bash
# Windows: Allow Docker in Windows Firewall
# Linux: Check iptables rules
sudo iptables -L -n | grep 8000

# Temporarily disable firewall for testing
# Windows: Turn off Windows Defender Firewall
# Linux: sudo ufw disable (re-enable after testing!)
```

---

## Production Deployment

### Security Hardening

1. **Use Strong Passwords**
   ```bash
   # Generate secure passwords
   openssl rand -base64 32

   # Update .env file
   POSTGRES_PASSWORD=$(openssl rand -base64 32)
   JWT_SECRET_KEY=$(openssl rand -hex 32)
   ADMIN_PASSWORD=$(openssl rand -base64 16)
   ```

2. **Enable HTTPS**
   - Use a reverse proxy (nginx, Traefik, Caddy)
   - Obtain SSL certificates (Let's Encrypt)
   - Update `ALLOWED_ORIGINS` to use HTTPS URLs

3. **Limit Exposed Ports**
   ```yaml
   # In docker-compose.yml, remove port mappings for PostgreSQL
   # Only expose API port through reverse proxy
   ```

4. **Use Docker Secrets** (Swarm mode)
   ```bash
   # Store sensitive data as Docker secrets
   echo "my_secret_password" | docker secret create postgres_password -
   ```

### Production Configuration

Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  postgres:
    # Use production-ready image
    image: postgres:14
    restart: always
    # Don't expose port to host
    ports: []
    # Enable logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  ml-api:
    # Use specific image tag
    image: ml-api:1.0.0
    restart: always
    # Production command with multiple workers
    command: gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 --timeout 300 main:app
    # Don't expose port to host (use reverse proxy)
    ports: []
    # Enable logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Deploy:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Monitoring and Logging

```bash
# View real-time logs
docker-compose logs -f

# Export logs
docker-compose logs > app.log

# Monitor resource usage
docker stats

# Set up log rotation (Linux)
sudo nano /etc/docker/daemon.json
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

### Backup Strategy

```bash
# Automated daily backups
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres pg_dump -U postgres -Fc ml_recommendation > backups/backup_$DATE.dump

# Keep only last 7 days
find backups/ -type f -mtime +7 -delete
```

**Add to crontab:**
```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

---

## Useful Tips

### Quick Reset (Development)

```bash
# Complete reset: stop, remove everything, start fresh
docker-compose down -v
docker-compose up -d
docker-compose exec ml-api python -m ml_api.scripts.migrate_all
```

### Development Workflow

```bash
# 1. Make code changes
# 2. Rebuild specific service
docker-compose build ml-api

# 3. Restart service
docker-compose restart ml-api

# 4. View logs
docker-compose logs -f ml-api
```

### Testing Inside Container

```bash
# Run tests
docker-compose exec ml-api pytest ml_api/tests/ -v

# Run specific test file
docker-compose exec ml-api pytest ml_api/tests/test_db_models.py -v

# Run with coverage
docker-compose exec ml-api pytest ml_api/tests/ --cov=ml_api --cov-report=html
```

---

## Next Steps

After successful setup:

1. ✅ Test API endpoints: `http://localhost:8000/docs`
2. ✅ Run database migration
3. ✅ Train ML models: `docker-compose exec ml-api python -m ml_api.scripts.retrain_models`
4. ✅ Set up monitoring and backups
5. ✅ Configure reverse proxy for production
6. ✅ Implement CI/CD pipeline

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review Docker logs: `docker-compose logs`
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)

---

**Last Updated:** January 23, 2026
