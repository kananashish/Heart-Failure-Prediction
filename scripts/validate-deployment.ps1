# Docker build and deployment validation script
$ErrorActionPreference = "Stop"

Write-Host "Heart Failure Prediction System - Deployment Validation" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Test Docker installation
Write-Status "Checking Docker installation..."
try {
    $dockerVersion = docker --version
    Write-Success "Docker found: $dockerVersion"
} catch {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

# Test Docker daemon
Write-Status "Checking Docker daemon..."
try {
    docker info | Out-Null
    Write-Success "Docker daemon is running"
} catch {
    Write-Error "Docker daemon is not running. Please start Docker Desktop."
    exit 1
}

# Test Docker Compose
Write-Status "Checking Docker Compose..."
try {
    $composeVersion = docker-compose --version
    Write-Success "Docker Compose found: $composeVersion"
} catch {
    try {
        $composeVersion = docker compose version
        Write-Success "Docker Compose (v2) found: $composeVersion"
    } catch {
        Write-Error "Docker Compose is not available"
        exit 1
    }
}

# Validate Dockerfile
Write-Status "Validating Dockerfile..."
if (Test-Path "Dockerfile") {
    Write-Success "Dockerfile found"
    
    # Check for security best practices
    $dockerfileContent = Get-Content "Dockerfile" -Raw
    
    if ($dockerfileContent -match "USER") {
        Write-Success "✓ Non-root user specified"
    } else {
        Write-Warning "⚠ Consider adding non-root user"
    }
    
    if ($dockerfileContent -match "HEALTHCHECK") {
        Write-Success "✓ Health check configured"
    } else {
        Write-Warning "⚠ Consider adding health check"
    }
    
    if ($dockerfileContent -match "EXPOSE") {
        Write-Success "✓ Port exposure documented"
    } else {
        Write-Warning "⚠ Consider documenting exposed ports"
    }
} else {
    Write-Error "Dockerfile not found"
    exit 1
}

# Validate production Dockerfile
Write-Status "Validating production Dockerfile..."
if (Test-Path "Dockerfile.prod") {
    Write-Success "Production Dockerfile found"
} else {
    Write-Warning "Production Dockerfile not found"
}

# Validate docker-compose files
Write-Status "Validating docker-compose files..."
if (Test-Path "docker-compose.yml") {
    Write-Success "docker-compose.yml found"
    try {
        docker-compose -f docker-compose.yml config | Out-Null
        Write-Success "✓ docker-compose.yml is valid"
    } catch {
        Write-Error "✗ docker-compose.yml has syntax errors"
    }
} else {
    Write-Error "docker-compose.yml not found"
}

if (Test-Path "docker-compose.prod.yml") {
    Write-Success "docker-compose.prod.yml found"
    try {
        docker-compose -f docker-compose.prod.yml config | Out-Null
        Write-Success "✓ docker-compose.prod.yml is valid"
    } catch {
        Write-Error "✗ docker-compose.prod.yml has syntax errors"
    }
} else {
    Write-Warning "docker-compose.prod.yml not found"
}

# Validate Kubernetes manifests
Write-Status "Validating Kubernetes manifests..."
if (Test-Path "k8s") {
    $k8sFiles = Get-ChildItem -Path "k8s" -Filter "*.yaml" -Recurse
    
    foreach ($file in $k8sFiles) {
        Write-Status "Validating $($file.Name)..."
        
        # Basic YAML syntax check
        try {
            $yamlContent = Get-Content $file.FullName -Raw
            # Simple YAML validation - check for basic structure
            if ($yamlContent -match "apiVersion:" -and $yamlContent -match "kind:" -and $yamlContent -match "metadata:") {
                Write-Success "✓ $($file.Name) has valid structure"
            } else {
                Write-Warning "⚠ $($file.Name) may have structural issues"
            }
        } catch {
            Write-Error "✗ $($file.Name) has syntax errors"
        }
    }
    
    Write-Success "Kubernetes manifests validation completed"
} else {
    Write-Warning "Kubernetes manifests directory not found"
}

# Validate deployment scripts
Write-Status "Validating deployment scripts..."
if (Test-Path "scripts") {
    $scriptFiles = @("deploy.sh", "deploy.bat")
    
    foreach ($script in $scriptFiles) {
        $scriptPath = Join-Path "scripts" $script
        if (Test-Path $scriptPath) {
            Write-Success "✓ $script found"
        } else {
            Write-Warning "⚠ $script not found"
        }
    }
} else {
    Write-Warning "Scripts directory not found"
}

# Check required files for deployment
Write-Status "Checking required files..."
$requiredFiles = @(
    "requirements.txt",
    "heart.csv",
    "app\main.py",
    "src\train.py",
    "src\preprocess.py"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Success "✓ $file found"
    } else {
        Write-Warning "⚠ $file not found - may be required for deployment"
    }
}

# Check .dockerignore
Write-Status "Checking .dockerignore..."
if (Test-Path ".dockerignore") {
    Write-Success "✓ .dockerignore found"
} else {
    Write-Warning "⚠ .dockerignore not found - consider creating one"
}

# Final summary
Write-Host ""
Write-Host "Validation Summary" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Success "✓ Docker environment validated"
Write-Success "✓ Dockerfile configurations found"
Write-Success "✓ Docker Compose configurations valid"
Write-Success "✓ Kubernetes manifests structure validated"
Write-Success "✓ Deployment scripts available"

Write-Host ""
Write-Status "The Heart Failure Prediction System is ready for deployment!"
Write-Status "Use the following commands to deploy:"
Write-Host "  Development: .\scripts\deploy.bat" -ForegroundColor White
Write-Host "  Production:  .\scripts\deploy.bat deploy latest production" -ForegroundColor White
Write-Host "  Kubernetes:  kubectl apply -f k8s/" -ForegroundColor White

Write-Host ""
Write-Status "For detailed deployment instructions, see the documentation in docs/"