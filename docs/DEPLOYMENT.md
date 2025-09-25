# Deployment Guide for Heart Failure Prediction System

## Overview

This guide provides comprehensive instructions for deploying the Heart Failure Prediction System across different environments using Docker, Docker Compose, and Kubernetes.

## Prerequisites

### Required Software
- **Docker Desktop**: Latest version (for Windows/macOS) or Docker Engine (for Linux)
- **Docker Compose**: v2.0 or later (usually included with Docker Desktop)
- **Kubernetes**: kubectl configured for your cluster
- **Git**: For version control and CI/CD
- **Python 3.9+**: For local development

### System Requirements
- **Minimum**: 2 CPU cores, 4GB RAM, 10GB disk space
- **Recommended**: 4 CPU cores, 8GB RAM, 50GB disk space
- **Production**: 8+ CPU cores, 16GB+ RAM, 100GB+ disk space

## Quick Start (Docker)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd heart-failure-prediction
```

### 2. Environment Configuration
Create a `.env` file:
```bash
# Application settings
MODEL_VERSION=1.0.0
LOG_LEVEL=INFO
DEBUG=false

# Database settings
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_NAME=heartfailure
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=postgres_password

# Redis settings
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

# Security
JWT_SECRET=your-jwt-secret-here
API_KEY=your-api-key-here
```

### 3. Deploy with Docker Compose

**Development Environment:**
```bash
# Windows
.\scripts\deploy.bat

# Linux/macOS
./scripts/deploy.sh
```

**Production Environment:**
```bash
# Windows
.\scripts\deploy.bat deploy latest production

# Linux/macOS
./scripts/deploy.sh deploy latest production
```

### 4. Access the Application
- **Application**: http://localhost:8501
- **Monitoring** (Production): http://localhost:3000 (admin/admin)
- **Metrics** (Production): http://localhost:9090

## Kubernetes Deployment

### 1. Prepare Kubernetes Environment

**Install kubectl** (if not already installed):
```bash
# Windows (using Chocolatey)
choco install kubernetes-cli

# macOS (using Homebrew)
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### 2. Configure Secrets
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Update secrets with your values
kubectl create secret generic heart-failure-prediction-secrets \
  --from-literal=database-password=your-db-password \
  --from-literal=redis-password=your-redis-password \
  --from-literal=api-key=your-api-key \
  --from-literal=jwt-secret=your-jwt-secret \
  -n heart-failure-prediction

# For private Docker registry
kubectl create secret docker-registry registry-secret \
  --docker-server=your-registry.com \
  --docker-username=your-username \
  --docker-password=your-password \
  --docker-email=your-email@example.com \
  -n heart-failure-prediction
```

### 3. Deploy Application
```bash
# Deploy core components
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy ingress (update domain names first)
kubectl apply -f k8s/ingress.yaml

# For production environments
kubectl apply -f k8s/production/
kubectl apply -f k8s/monitoring/
```

### 4. Verify Deployment
```bash
# Check pod status
kubectl get pods -n heart-failure-prediction

# Check service status
kubectl get services -n heart-failure-prediction

# View logs
kubectl logs -f deployment/heart-failure-prediction -n heart-failure-prediction

# Check ingress
kubectl get ingress -n heart-failure-prediction
```

## Cloud Platform Specific Instructions

### Amazon Web Services (EKS)

**1. Setup EKS Cluster:**
```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster --name heart-failure-cluster --region us-west-2 --nodegroup-name standard-workers --node-type t3.medium --nodes 3
```

**2. Configure LoadBalancer:**
```bash
# Install AWS Load Balancer Controller
kubectl apply -f https://github.com/kubernetes-sigs/aws-load-balancer-controller/releases/download/v2.4.0/v2_4_0_full.yaml
```

**3. Update ingress for ALB:**
```yaml
# In k8s/ingress.yaml, uncomment AWS ALB annotations
kubernetes.io/ingress.class: alb
alb.ingress.kubernetes.io/scheme: internet-facing
alb.ingress.kubernetes.io/target-type: ip
```

### Microsoft Azure (AKS)

**1. Setup AKS Cluster:**
```bash
# Create resource group
az group create --name heart-failure-rg --location eastus

# Create AKS cluster
az aks create --resource-group heart-failure-rg --name heart-failure-cluster --node-count 3 --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group heart-failure-rg --name heart-failure-cluster
```

**2. Configure Application Gateway:**
```bash
# Enable Application Gateway Ingress Controller
az aks enable-addons -n heart-failure-cluster -g heart-failure-rg -a ingress-appgw --appgw-name heart-failure-appgw --appgw-subnet-cidr "10.2.0.0/16"
```

### Google Cloud Platform (GKE)

**1. Setup GKE Cluster:**
```bash
# Create cluster
gcloud container clusters create heart-failure-cluster --zone us-central1-a --num-nodes 3

# Get credentials
gcloud container clusters get-credentials heart-failure-cluster --zone us-central1-a
```

**2. Configure Ingress:**
```bash
# Enable GCE ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-gce/master/deploy/static/rbac.yaml
```

## Monitoring and Observability

### Prometheus and Grafana Setup
The system includes comprehensive monitoring:

**1. Deploy monitoring stack:**
```bash
kubectl apply -f k8s/monitoring/
```

**2. Access dashboards:**
- **Grafana**: http://your-domain/grafana (admin/admin)
- **Prometheus**: http://your-domain/prometheus

**3. Configure alerts:**
```yaml
# Create AlertManager configuration
kubectl create configmap alertmanager-config \
  --from-file=alertmanager.yml=monitoring/alertmanager.yml \
  -n heart-failure-prediction
```

### Log Management
**1. Configure log aggregation:**
```bash
# Deploy Fluent Bit for log collection
kubectl apply -f monitoring/fluent-bit.yaml

# Configure log forwarding to your preferred system
# (ELK Stack, Splunk, DataDog, etc.)
```

## Security Considerations

### 1. Network Security
```bash
# Apply network policies
kubectl apply -f k8s/production/network-policy.yaml
```

### 2. Secret Management
- Use external secret management (AWS Secrets Manager, Azure Key Vault, etc.)
- Rotate secrets regularly
- Implement least privilege access

### 3. Image Security
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image heart-failure-prediction:latest
```

## Scaling and Performance

### Horizontal Pod Autoscaler
```bash
# Apply HPA configuration
kubectl apply -f k8s/production/hpa.yaml

# Monitor scaling
kubectl get hpa -n heart-failure-prediction
```

### Resource Optimization
1. **CPU/Memory tuning**: Adjust resource requests/limits based on monitoring
2. **Storage optimization**: Use appropriate storage classes
3. **Network optimization**: Configure ingress controllers for performance

## Backup and Disaster Recovery

### 1. Data Backup
```bash
# Backup persistent volumes
kubectl exec -n heart-failure-prediction deployment/heart-failure-prediction -- \
  tar czf /tmp/models-backup.tar.gz /app/models

# Copy backup to external storage
kubectl cp heart-failure-prediction/pod-name:/tmp/models-backup.tar.gz ./models-backup.tar.gz
```

### 2. Configuration Backup
```bash
# Export all configurations
kubectl get all,configmaps,secrets,pvc -n heart-failure-prediction -o yaml > backup.yaml
```

### 3. Database Backup
```bash
# PostgreSQL backup (if using external database)
kubectl exec -n heart-failure-prediction postgres-pod -- \
  pg_dump -U postgres heartfailure > database-backup.sql
```

## Troubleshooting

### Common Issues

**1. Pod fails to start:**
```bash
kubectl describe pod pod-name -n heart-failure-prediction
kubectl logs pod-name -n heart-failure-prediction
```

**2. Service not accessible:**
```bash
# Check service endpoints
kubectl get endpoints -n heart-failure-prediction

# Test internal connectivity
kubectl run debug --image=busybox -it --rm --restart=Never -- \
  wget -qO- http://heart-failure-prediction-service:8501/_stcore/health
```

**3. Ingress not working:**
```bash
# Check ingress status
kubectl describe ingress -n heart-failure-prediction

# Verify DNS resolution
nslookup your-domain.com
```

**4. Performance issues:**
```bash
# Check resource usage
kubectl top pods -n heart-failure-prediction
kubectl top nodes

# Monitor metrics
kubectl port-forward -n heart-failure-prediction svc/prometheus-service 9090:9090
```

### Log Analysis
```bash
# Stream logs from all pods
kubectl logs -f -l app=heart-failure-prediction -n heart-failure-prediction

# Check specific error patterns
kubectl logs -n heart-failure-prediction deployment/heart-failure-prediction | grep ERROR
```

## Maintenance and Updates

### Rolling Updates
```bash
# Update image version
kubectl set image deployment/heart-failure-prediction \
  heart-failure-prediction=heart-failure-prediction:v1.1.0 \
  -n heart-failure-prediction

# Monitor rollout
kubectl rollout status deployment/heart-failure-prediction -n heart-failure-prediction

# Rollback if needed
kubectl rollout undo deployment/heart-failure-prediction -n heart-failure-prediction
```

### Health Checks
```bash
# Manual health verification
curl -f http://your-domain/_stcore/health
curl -f http://your-domain/metrics

# Automated health monitoring
kubectl get pods -n heart-failure-prediction -w
```

## Support and Contact

For deployment issues:
1. Check the logs first: `kubectl logs -n heart-failure-prediction deployment/heart-failure-prediction`
2. Review monitoring dashboards
3. Consult troubleshooting section
4. Contact system administrators

---

**Note**: This guide assumes familiarity with Docker and Kubernetes concepts. For production deployments, consider consulting with DevOps professionals to ensure optimal configuration for your specific environment.