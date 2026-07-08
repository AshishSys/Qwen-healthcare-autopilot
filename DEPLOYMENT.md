# Deployment Guide — Alibaba Cloud

## Prerequisites

1. **Alibaba Cloud Account** — [Sign up](https://account.alibabacloud.com/register/intl_register.htm)
2. **DashScope API Key** — [Get key](https://dashscope.console.aliyun.com/apiKey)
3. **Alibaba Cloud CLI (aliyun)** — [Install](https://www.alibabacloud.com/help/en/cli/install)
4. **Docker** installed locally
5. **Git** for repository management

---

## Step 1: Push to GitHub

```bash
# Initialize repo
cd healthcare-autopilot
git init
git add .
git commit -m "feat: Healthcare Autopilot Agent - autonomous triage system"

# Create GitHub repo (use GitHub CLI or web UI)
gh repo create healthcare-autopilot --public --source=. --push

# Verify license is visible at top of repo page
# GitHub auto-detects Apache-2.0 from LICENSE file
```

---

## Step 2: Set Up Alibaba Cloud Resources

### 2.1 Create ECS Instance

```bash
# Login to Alibaba Cloud CLI
aliyun configure

# Create a VPC
aliyun vpc CreateVpc \
  --RegionId us-east-1 \
  --VpcName healthcare-autopilot-vpc \
  --CidrBlock 172.16.0.0/16

# Create ECS Instance (2 vCPU, 4GB — sufficient for demo)
aliyun ecs CreateInstance \
  --RegionId us-east-1 \
  --InstanceType ecs.g7.large \
  --ImageId ubuntu_22_04_x64_20G_alibase_20240101.vhd \
  --InstanceName healthcare-autopilot \
  --InternetMaxBandwidthOut 5 \
  --SecurityGroupId <your-sg-id> \
  --VSwitchId <your-vswitch-id>
```

### 2.2 Create RDS PostgreSQL (Optional for demo — can use SQLite)

```bash
aliyun rds CreateDBInstance \
  --RegionId us-east-1 \
  --Engine PostgreSQL \
  --EngineVersion 15.0 \
  --DBInstanceClass rds.pg.s1.small \
  --DBInstanceStorage 20 \
  --DBInstanceNetType Intranet \
  --PayType Postpaid
```

### 2.3 Create OSS Bucket

```bash
aliyun oss mb oss://healthcare-autopilot-docs --region us-east-1
```

---

## Step 3: Deploy with Docker

### 3.1 Build & Push to Alibaba Cloud Container Registry (ACR)

```bash
# Login to ACR
docker login --username=<your-username> registry.us-east-1.cr.aliyuncs.com

# Build image
docker build -t healthcare-autopilot:latest .

# Tag for ACR
docker tag healthcare-autopilot:latest \
  registry.us-east-1.cr.aliyuncs.com/healthcare/autopilot:latest

# Push
docker push registry.us-east-1.cr.aliyuncs.com/healthcare/autopilot:latest
```

### 3.2 Deploy on ECS

SSH into your ECS instance and run:

```bash
# Install Docker on ECS
sudo apt update && sudo apt install -y docker.io docker-compose
sudo systemctl enable docker && sudo systemctl start docker

# Pull and run
docker pull registry.us-east-1.cr.aliyuncs.com/healthcare/autopilot:latest

docker run -d \
  --name healthcare-autopilot \
  --restart unless-stopped \
  -p 80:8000 \
  -e DASHSCOPE_API_KEY="your-dashscope-api-key" \
  -e RDS_CONNECTION_STRING="postgresql://user:pass@pgm-xxx.pg.rds.aliyuncs.com:5432/healthcare" \
  -e ALIBABA_CLOUD_ACCESS_KEY_ID="your-key-id" \
  -e ALIBABA_CLOUD_ACCESS_KEY_SECRET="your-key-secret" \
  registry.us-east-1.cr.aliyuncs.com/healthcare/autopilot:latest
```

### 3.3 Verify Deployment

```bash
# Health check
curl http://<your-ecs-public-ip>/health

# Expected response:
# {"status":"healthy","qwen_cloud":"connected","active_workflows":0}
```

---

## Step 4: Quick Deploy Script (One-Command)

Use the provided `deploy.sh`:

```bash
chmod +x deployment/deploy.sh
./deployment/deploy.sh
```

---

## Step 5: Configure Security Group

Allow inbound traffic on port 80 (HTTP) and 443 (HTTPS):

```bash
aliyun ecs AuthorizeSecurityGroup \
  --RegionId us-east-1 \
  --SecurityGroupId <sg-id> \
  --IpProtocol tcp \
  --PortRange 80/80 \
  --SourceCidrIp 0.0.0.0/0

aliyun ecs AuthorizeSecurityGroup \
  --RegionId us-east-1 \
  --SecurityGroupId <sg-id> \
  --IpProtocol tcp \
  --PortRange 443/443 \
  --SourceCidrIp 0.0.0.0/0
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DASHSCOPE_API_KEY` | ✅ | Qwen Cloud API key from DashScope console |
| `RDS_CONNECTION_STRING` | ❌ | PostgreSQL connection (optional for demo) |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` | ❌ | For OSS access |
| `ALIBABA_CLOUD_ACCESS_KEY_SECRET` | ❌ | For OSS access |
| `REDIS_URL` | ❌ | Redis connection for session state |

---

## Cost Estimate (Demo/Hackathon)

| Service | Spec | Monthly Cost |
|---------|------|-------------|
| ECS | ecs.g7.large (2 vCPU, 8 GB) | ~$30 |
| RDS | rds.pg.s1.small (1 vCPU, 2 GB) | ~$15 |
| OSS | 5 GB storage | ~$0.12 |
| DashScope | Pay-per-token (Qwen-max) | ~$5-20 |
| **Total** | | **~$50-65/month** |

*Free tier available for new Alibaba Cloud accounts.*

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DASHSCOPE_API_KEY not set` | Set env var: `export DASHSCOPE_API_KEY=sk-xxx` |
| Connection timeout to DashScope | Check ECS security group allows outbound HTTPS |
| Docker build fails | Ensure Python 3.11+ base image |
| Port 80 not accessible | Configure security group inbound rules |
