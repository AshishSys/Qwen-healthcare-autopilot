#!/bin/bash
# ============================================================
# Healthcare Autopilot — Alibaba Cloud Deployment Script
# ============================================================
# Usage: ./deployment/deploy.sh
#
# Prerequisites:
#   - Docker installed
#   - Alibaba Cloud CLI configured (aliyun configure)
#   - DASHSCOPE_API_KEY environment variable set
# ============================================================

set -e

echo "🏥 Healthcare Autopilot — Alibaba Cloud Deployment"
echo "=================================================="

# Configuration
REGION="us-east-1"
REGISTRY="registry.${REGION}.cr.aliyuncs.com"
NAMESPACE="healthcare"
IMAGE_NAME="autopilot"
TAG="latest"
FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
CONTAINER_NAME="healthcare-autopilot"

# ---- Validation ----
echo ""
echo "📋 Validating prerequisites..."

if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "❌ DASHSCOPE_API_KEY not set!"
    echo "   Get your key at: https://dashscope.console.aliyun.com/apiKey"
    echo "   Then run: export DASHSCOPE_API_KEY=sk-your-key"
    exit 1
fi
echo "   ✅ DASHSCOPE_API_KEY is set"

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install Docker first."
    exit 1
fi
echo "   ✅ Docker available"

# ---- Build ----
echo ""
echo "🔨 Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .
echo "   ✅ Image built: ${IMAGE_NAME}:${TAG}"

# ---- Tag & Push ----
echo ""
echo "📤 Tagging and pushing to Alibaba Cloud Container Registry..."
docker tag ${IMAGE_NAME}:${TAG} ${FULL_IMAGE}

echo "   Pushing to: ${FULL_IMAGE}"
docker push ${FULL_IMAGE} 2>/dev/null || {
    echo ""
    echo "⚠️  Push failed. You may need to:"
    echo "   1. Create namespace '${NAMESPACE}' in ACR console"
    echo "   2. Login: docker login ${REGISTRY}"
    echo ""
    echo "   Continuing with local deployment..."
}

# ---- Deploy Locally (or on ECS via SSH) ----
echo ""
echo "🚀 Deploying container..."

# Stop existing container if running
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Run
docker run -d \
  --name ${CONTAINER_NAME} \
  --restart unless-stopped \
  -p 8000:8000 \
  -e DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY}" \
  -e QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  ${IMAGE_NAME}:${TAG}

echo "   ✅ Container started: ${CONTAINER_NAME}"

# ---- Health Check ----
echo ""
echo "🏥 Waiting for service to start..."
sleep 3

HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "")
if echo "$HEALTH" | grep -q "healthy"; then
    echo "   ✅ Service is healthy!"
    echo ""
    echo "=================================================="
    echo "🎉 Deployment successful!"
    echo ""
    echo "   API:    http://localhost:8000"
    echo "   Health: http://localhost:8000/health"
    echo "   Docs:   http://localhost:8000/docs"
    echo ""
    echo "   Test triage:"
    echo "   curl -X POST http://localhost:8000/api/v1/triage \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"chief_complaint\":\"severe headache\",\"symptoms\":[\"headache\",\"nausea\"],\"duration\":\"2 days\",\"severity\":7}'"
    echo "=================================================="
else
    echo "   ⚠️  Service may still be starting. Check logs:"
    echo "   docker logs ${CONTAINER_NAME}"
fi
