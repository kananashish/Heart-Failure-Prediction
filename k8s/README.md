# Kubernetes Deployment Configuration for Heart Failure Prediction System

## Overview
This directory contains Kubernetes manifests for deploying the Heart Failure Prediction System in a Kubernetes cluster.

## Files Structure
```
k8s/
├── namespace.yaml          # Application namespace
├── configmap.yaml          # Configuration data
├── secret.yaml             # Sensitive data (passwords, tokens)
├── deployment.yaml         # Application deployment
├── service.yaml            # Service definition
├── ingress.yaml            # Ingress for external access
├── persistent-volume.yaml  # Data storage
├── monitoring/             # Monitoring stack
│   ├── prometheus.yaml     # Prometheus metrics
│   └── grafana.yaml        # Grafana dashboards
└── production/             # Production-specific configs
    ├── hpa.yaml            # Horizontal Pod Autoscaler
    └── network-policy.yaml # Network security policies
```

## Prerequisites
- Kubernetes cluster (v1.20+)
- kubectl configured to access your cluster
- Docker images built and pushed to a registry

## Quick Start

1. **Create namespace and secrets:**
   ```bash
   kubectl apply -f namespace.yaml
   kubectl apply -f secret.yaml
   ```

2. **Deploy configuration and storage:**
   ```bash
   kubectl apply -f configmap.yaml
   kubectl apply -f persistent-volume.yaml
   ```

3. **Deploy the application:**
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

4. **Configure external access:**
   ```bash
   kubectl apply -f ingress.yaml
   ```

## Production Deployment

For production environments, also apply:
```bash
kubectl apply -f production/
kubectl apply -f monitoring/
```

## Configuration

### Environment Variables
Configure the following in `configmap.yaml`:
- `MODEL_VERSION`: Model version to use
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- `MAX_WORKERS`: Number of worker processes

### Secrets
Configure the following in `secret.yaml`:
- Database passwords
- API keys
- SSL certificates

### Resources
Default resource limits:
- CPU: 500m request, 1000m limit
- Memory: 512Mi request, 1Gi limit

Adjust these based on your requirements in `deployment.yaml`.

## Scaling

### Manual Scaling
```bash
kubectl scale deployment heart-failure-prediction --replicas=3
```

### Auto Scaling
The HPA (Horizontal Pod Autoscaler) is configured for production:
- Min replicas: 2
- Max replicas: 10
- Target CPU: 70%

## Monitoring

Access monitoring services:
- Prometheus: `http://your-domain/prometheus`
- Grafana: `http://your-domain/grafana` (admin/admin)

## Health Checks

The application includes:
- Liveness probe: `/health`
- Readiness probe: `/_stcore/health`

## Storage

Persistent volumes are configured for:
- Model artifacts: `/app/models`
- Application logs: `/app/logs`
- User data: `/app/data`

## Security

Production deployments include:
- Network policies for pod communication
- RBAC for service accounts
- SecurityContext for containers
- Secret management for sensitive data

## Troubleshooting

### Check pod status:
```bash
kubectl get pods -n heart-failure-prediction
```

### View logs:
```bash
kubectl logs -f deployment/heart-failure-prediction -n heart-failure-prediction
```

### Debug pod issues:
```bash
kubectl describe pod <pod-name> -n heart-failure-prediction
```

### Access pod shell:
```bash
kubectl exec -it <pod-name> -n heart-failure-prediction -- /bin/bash
```

## Cleanup

To remove all resources:
```bash
kubectl delete -f .
kubectl delete namespace heart-failure-prediction
```

## Support

For issues and questions:
1. Check pod logs and events
2. Verify configuration in ConfigMaps and Secrets
3. Check resource utilization
4. Review network policies and ingress configuration