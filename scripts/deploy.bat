@echo off
REM Windows Docker deployment script for Heart Failure Prediction System

setlocal enabledelayedexpansion

REM Configuration
set APP_NAME=heart-failure-prediction
set VERSION=%1
set ENVIRONMENT=%2

if "%VERSION%"=="" set VERSION=latest
if "%ENVIRONMENT%"=="" set ENVIRONMENT=development

echo Heart Failure Prediction System - Docker Deployment
echo ==================================================

REM Function to print status
:print_status
echo [INFO] %1
goto :eof

:print_warning  
echo [WARNING] %1
goto :eof

:print_error
echo [ERROR] %1
goto :eof

REM Check if Docker is installed and running
:check_docker
call :print_status "Checking Docker installation..."
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not running. Please start Docker Desktop."
    exit /b 1
)

call :print_status "Docker is installed and running"
goto :eof

REM Check if Docker Compose is available
:check_docker_compose
call :print_status "Checking Docker Compose..."
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    docker compose version >nul 2>&1
    if !errorlevel! neq 0 (
        call :print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit /b 1
    )
)
call :print_status "Docker Compose is available"
goto :eof

REM Build the Docker image
:build_image
call :print_status "Building Docker image..."

if "%ENVIRONMENT%"=="production" (
    docker build -f Dockerfile.prod -t %APP_NAME%:%VERSION% .
) else (
    docker build -t %APP_NAME%:%VERSION% .
)

if %errorlevel% neq 0 (
    call :print_error "Failed to build Docker image"
    exit /b 1
)

call :print_status "Docker image built successfully"
goto :eof

REM Create necessary directories
:create_directories
call :print_status "Creating necessary directories..."
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "secrets" mkdir secrets

REM Create dummy secrets for development
if "%ENVIRONMENT%"=="development" (
    echo dev_db_password > secrets\db_password.txt
    echo dev_redis_password > secrets\redis_password.txt
    echo admin > secrets\grafana_password.txt
)

call :print_status "Directories created"
goto :eof

REM Setup data
:setup_data
call :print_status "Setting up data..."
if not exist "data\heart.csv" (
    call :print_warning "heart.csv not found in data\. Please download the dataset."
    call :print_status "You can download it from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"
) else (
    call :print_status "Data files found"
)
goto :eof

REM Start services
:start_services
call :print_status "Starting services..."

if "%ENVIRONMENT%"=="production" (
    docker-compose -f docker-compose.prod.yml up -d
) else (
    docker-compose up -d
)

if %errorlevel% neq 0 (
    call :print_error "Failed to start services"
    exit /b 1
)

call :print_status "Services started"
goto :eof

REM Wait for services to be ready
:wait_for_services
call :print_status "Waiting for services to be ready..."

set counter=0
set timeout=60

:health_check_loop
if %counter% geq %timeout% goto health_check_timeout

REM Use PowerShell to check health endpoint
powershell -Command "try { Invoke-WebRequest -Uri http://localhost:8501/_stcore/health -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel% equ 0 (
    call :print_status "Heart Failure Prediction app is ready"
    goto :eof
)

timeout /t 2 /nobreak >nul
set /a counter+=2
goto health_check_loop

:health_check_timeout
call :print_warning "App health check timed out, but it might still be starting..."
goto :eof

REM Show running services
:show_services
call :print_status "Running services:"
if "%ENVIRONMENT%"=="production" (
    docker-compose -f docker-compose.prod.yml ps
) else (
    docker-compose ps
)
goto :eof

REM Show access URLs
:show_urls
echo.
call :print_status "Access URLs:"
echo   Heart Failure Prediction App: http://localhost:8501

if "%ENVIRONMENT%"=="production" (
    echo   Monitoring (Grafana): http://localhost:3000 (admin/admin)
    echo   Metrics (Prometheus): http://localhost:9090
)
echo.
goto :eof

REM Main deployment flow
:main
call :print_status "Starting deployment for environment: %ENVIRONMENT%"

call :check_docker
if %errorlevel% neq 0 exit /b 1

call :check_docker_compose
if %errorlevel% neq 0 exit /b 1

call :create_directories
call :setup_data
call :build_image
if %errorlevel% neq 0 exit /b 1

call :start_services
if %errorlevel% neq 0 exit /b 1

call :wait_for_services
call :show_services
call :show_urls

call :print_status "Deployment completed successfully!"
call :print_status "The Heart Failure Prediction System is now running."
goto :eof

REM Handle script arguments
if "%1"=="stop" (
    call :print_status "Stopping services..."
    if "%ENVIRONMENT%"=="production" (
        docker-compose -f docker-compose.prod.yml down
    ) else (
        docker-compose down
    )
    call :print_status "Services stopped"
    exit /b 0
)

if "%1"=="restart" (
    call :print_status "Restarting services..."
    if "%ENVIRONMENT%"=="production" (
        docker-compose -f docker-compose.prod.yml restart
    ) else (
        docker-compose restart
    )
    call :print_status "Services restarted"
    exit /b 0
)

if "%1"=="logs" (
    call :print_status "Showing logs..."
    if "%ENVIRONMENT%"=="production" (
        docker-compose -f docker-compose.prod.yml logs -f
    ) else (
        docker-compose logs -f
    )
    exit /b 0
)

if "%1"=="clean" (
    call :print_status "Cleaning up..."
    if "%ENVIRONMENT%"=="production" (
        docker-compose -f docker-compose.prod.yml down -v --remove-orphans
    ) else (
        docker-compose down -v --remove-orphans
    )
    docker system prune -f
    call :print_status "Cleanup completed"
    exit /b 0
)

if "%1"=="help" (
    echo Usage: %0 [COMMAND] [VERSION] [ENVIRONMENT]
    echo.
    echo Commands:
    echo   deploy     Deploy the application (default)
    echo   stop       Stop all services
    echo   restart    Restart all services
    echo   logs       Show service logs
    echo   clean      Clean up containers and volumes
    echo   help       Show this help message
    echo.
    echo Environment: development (default) | production
    echo Version: latest (default) | specific version tag
    echo.
    echo Examples:
    echo   %0                           # Deploy latest in development
    echo   %0 deploy latest production  # Deploy latest in production
    echo   %0 stop                      # Stop all services
    echo   %0 logs                      # Show logs
    exit /b 0
)

if not "%1"=="deploy" if not "%1"=="" (
    call :print_error "Unknown command: %1"
    call :print_status "Use '%0 help' to see available commands"
    exit /b 1
)

REM Run main deployment
call :main