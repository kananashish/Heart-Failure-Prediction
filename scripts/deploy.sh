#!/bin/bash
# Docker deployment script for Heart Failure Prediction System

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="heart-failure-prediction"
VERSION=${1:-latest}
ENVIRONMENT=${2:-development}

echo -e "${BLUE}Heart Failure Prediction System - Docker Deployment${NC}"
echo -e "${BLUE}=================================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker daemon."
        exit 1
    fi
    
    print_status "Docker is installed and running ✓"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose is available ✓"
}

# Build the Docker image
build_image() {
    print_status "Building Docker image..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker build -f Dockerfile.prod -t $APP_NAME:$VERSION .
    else
        docker build -t $APP_NAME:$VERSION .
    fi
    
    print_status "Docker image built successfully ✓"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data models logs secrets
    
    # Create dummy secrets if they don't exist (for development)
    if [ "$ENVIRONMENT" = "development" ]; then
        echo "dev_db_password" > secrets/db_password.txt || true
        echo "dev_redis_password" > secrets/redis_password.txt || true
        echo "admin" > secrets/grafana_password.txt || true
    fi
    
    print_status "Directories created ✓"
}

# Download sample data if not exists
setup_data() {
    print_status "Setting up data..."
    if [ ! -f "data/heart.csv" ]; then
        print_warning "heart.csv not found in data/. Please download the dataset."
        print_status "You can download it from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"
    else
        print_status "Data files found ✓"
    fi
}

# Start services
start_services() {
    print_status "Starting services..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    print_status "Services started ✓"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for main app
    timeout=60
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            print_status "Heart Failure Prediction app is ready ✓"
            break
        fi
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ $counter -ge $timeout ]; then
        print_warning "App health check timed out, but it might still be starting..."
    fi
}

# Show running services
show_services() {
    print_status "Running services:"
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.prod.yml ps
    else
        docker-compose ps
    fi
}

# Show access URLs
show_urls() {
    echo ""
    print_status "Access URLs:"
    echo -e "  ${BLUE}Heart Failure Prediction App:${NC} http://localhost:8501"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo -e "  ${BLUE}Monitoring (Grafana):${NC} http://localhost:3000 (admin/admin)"
        echo -e "  ${BLUE}Metrics (Prometheus):${NC} http://localhost:9090"
    fi
    
    echo ""
}

# Main deployment flow
main() {
    print_status "Starting deployment for environment: $ENVIRONMENT"
    
    check_docker
    check_docker_compose
    create_directories
    setup_data
    build_image
    start_services
    wait_for_services
    show_services
    show_urls
    
    print_status "Deployment completed successfully! 🎉"
    print_status "The Heart Failure Prediction System is now running."
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping services..."
        if [ "$ENVIRONMENT" = "production" ]; then
            docker-compose -f docker-compose.prod.yml down
        else
            docker-compose down
        fi
        print_status "Services stopped ✓"
        exit 0
        ;;
    "restart")
        print_status "Restarting services..."
        if [ "$ENVIRONMENT" = "production" ]; then
            docker-compose -f docker-compose.prod.yml restart
        else
            docker-compose restart
        fi
        print_status "Services restarted ✓"
        exit 0
        ;;
    "logs")
        print_status "Showing logs..."
        if [ "$ENVIRONMENT" = "production" ]; then
            docker-compose -f docker-compose.prod.yml logs -f
        else
            docker-compose logs -f
        fi
        exit 0
        ;;
    "clean")
        print_status "Cleaning up..."
        if [ "$ENVIRONMENT" = "production" ]; then
            docker-compose -f docker-compose.prod.yml down -v --remove-orphans
        else
            docker-compose down -v --remove-orphans
        fi
        docker system prune -f
        print_status "Cleanup completed ✓"
        exit 0
        ;;
    "help")
        echo "Usage: $0 [COMMAND] [VERSION] [ENVIRONMENT]"
        echo ""
        echo "Commands:"
        echo "  deploy     Deploy the application (default)"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  logs       Show service logs"
        echo "  clean      Clean up containers and volumes"
        echo "  help       Show this help message"
        echo ""
        echo "Environment: development (default) | production"
        echo "Version: latest (default) | specific version tag"
        echo ""
        echo "Examples:"
        echo "  $0                           # Deploy latest in development"
        echo "  $0 deploy latest production  # Deploy latest in production"
        echo "  $0 stop                      # Stop all services"
        echo "  $0 logs                      # Show logs"
        exit 0
        ;;
    "deploy"|"")
        # Continue to main deployment
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Use '$0 help' to see available commands"
        exit 1
        ;;
esac

# Run main deployment
main