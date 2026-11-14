# deploy.ps1
# PowerShell automated deployment script for CoralGuard on Google Cloud Run.
# Usage:
#   .\deploy.ps1 -ProjectId "your-gcp-project-id" -Region "us-central1"

param (
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    [string]$Region = "us-central1"
)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "   CORALGUARD: GOOGLE CLOUD RUN AUTOMATED DEPLOYMENT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Project ID: $ProjectId"
Write-Host "Region:     $Region"
Write-Host ""

# 1. Set Active GCP Project
Write-Host ">>> Step 1/3: Setting active GCP project..." -ForegroundColor Yellow
gcloud config set project $ProjectId

# 2. Build Container Image via Cloud Build
$ImageTag = "gcr.io/$ProjectId/coralguard:latest"
Write-Host ">>> Step 2/3: Building and pushing container image ($ImageTag)..." -ForegroundColor Yellow
gcloud builds submit --tag $ImageTag .

# 3. Deploy Dashboard to Cloud Run
Write-Host ">>> Step 3a/3: Deploying CoralGuard Dashboard to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy coralguard-dashboard `
    --image $ImageTag `
    --platform managed `
    --region $Region `
    --port 8501 `
    --memory 2Gi `
    --cpu 2 `
    --allow-unauthenticated

# 4. Deploy REST API to Cloud Run
Write-Host ">>> Step 3b/3: Deploying CoralGuard REST API to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy coralguard-api `
    --image $ImageTag `
    --command "python" `
    --args "-m,src.server,--port,8000" `
    --platform managed `
    --region $Region `
    --port 8000 `
    --memory 1Gi `
    --cpu 1 `
    --allow-unauthenticated

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "   DEPLOYMENT COMPLETE! Services are now live on Google Cloud Run." -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green

