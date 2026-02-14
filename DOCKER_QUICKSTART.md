# Docker Quick Start Guide

Fast setup guide for running the ML Recommendation API with Docker.

---

## 🚀 Quick Setup (5 Minutes)

```bash
# 1. Create environment file
cp .env.docker.template .env

# 2. Edit .env - REQUIRED! Set these values:
#    - POSTGRES_PASSWORD=your_password
#    - JWT_SECRET_KEY=your_secret_key (generate: openssl rand -hex 32)
#    - ADMIN_PASSWORD=your_admin_password

# 3. Start services
docker-compose up -d

# 4. Run database migration (wait 30 seconds first)
docker-compose exec ml-api python -m ml_api.scripts.migrate_all

# 5. Verify it's working
curl http://localhost:8000/health
```

**✅ Done! API is running at http://localhost:8000**

---

## 📌 Essential Commands

### Service Control
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart API only
docker-compose restart ml-api

# View status
docker-compose ps
```

### Logs
```bash
# View all logs
docker-compose logs -f

# View API logs only
docker-compose logs -f ml-api

# View PostgreSQL logs only
docker-compose logs -f postgres
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d ml_recommendation

# Common queries:
# SELECT COUNT(*) FROM products;
# SELECT COUNT(*) FROM ratings;
# \dt          -- List tables
# \q           -- Quit
```

### Execute Commands
```bash
# Run migration
docker-compose exec ml-api python -m ml_api.scripts.migrate_all

# Retrain models
docker-compose exec ml-api python -m ml_api.scripts.retrain_models

# Run tests
docker-compose exec ml-api pytest ml_api/tests/ -v

# Open bash shell
docker-compose exec ml-api bash
```

---

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | Main API |
| **Swagger Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Health Check** | http://localhost:8000/health | API status |
| **PostgreSQL** | localhost:5432 | Database (use GUI client) |

---

## 🔧 Troubleshooting

### API won't start
```bash
# Check logs
docker-compose logs ml-api

# Common issue: Port already in use
# Fix: Change API_PORT=8001 in .env
```

### Migration fails
```bash
# Ensure CSV files exist
ls ratings_Beauty.csv
ls content/product_descriptions.csv

# Restart and retry
docker-compose restart
docker-compose exec ml-api python -m ml_api.scripts.migrate_all
```

### Database connection error
```bash
# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready -U postgres

# If not ready, restart
docker-compose restart postgres
```

### Reset everything
```bash
# WARNING: Deletes all data!
docker-compose down -v
docker-compose up -d
docker-compose exec ml-api python -m ml_api.scripts.migrate_all
```

---

## 📊 Testing the API

### Login as Admin
```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=YOUR_ADMIN_PASSWORD"

# Save the access_token from response
export TOKEN="your_token_here"
```

### Test Endpoints
```bash
# Get popular products
curl -X POST http://localhost:8000/recommend/popular \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"top_n": 5}'

# List products
curl http://localhost:8000/recommend/products?limit=5 \
  -H "Authorization: Bearer $TOKEN"

# List categories
curl http://localhost:8000/recommend/categories \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🛑 Shutdown

```bash
# Stop services (keeps data)
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v
```

---

## 📚 Full Documentation

For detailed instructions, see **[DOCKER_SETUP.md](./DOCKER_SETUP.md)**

---

**Need Help?** Check `docker-compose logs` for errors
