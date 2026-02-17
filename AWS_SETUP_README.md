# AWS Integration Setup for AAC
## Prerequisites

1. **AWS Account**: Create an AWS account at https://aws.amazon.com
2. **AWS CLI**: Install AWS CLI v2 from https://aws.amazon.com/cli/
3. **Configure AWS Credentials**:
   ```bash
   aws configure
   ```
   Or set environment variables in `.env`:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-1
   ```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your AWS credentials:
   ```bash
   cp src/aac/config/.env.example .env
   # Edit .env with your values
   ```

3. **Deploy to AWS**:
   ```bash
   python aws_deployment.py --deploy
   ```

4. **Check Status**:
   ```bash
   python aws_deployment.py --status
   ```

5. **Backup Data**:
   ```bash
   python aws_deployment.py --backup
   ```

## What Gets Deployed

- **EC2 Instance**: t3.micro (free tier eligible) with Ubuntu 22.04
- **Security Group**: Open ports for SSH (22), HTTP (80), HTTPS (443), Dash (8050), Streamlit (8081)
- **S3 Bucket**: For data backups and storage
- **CloudWatch**: Monitoring and alerts
- **KMS**: Encryption for sensitive data (optional)

## Manual Configuration Steps

After deployment, SSH into your EC2 instance and complete setup:

```bash
# SSH into instance (replace with your instance IP)
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.11 python3.11-venv python3-pip git htop tmux -y

# Clone and setup AAC
git clone https://github.com/your-repo/aac-system.git
cd aac-system
python3 -m venv aac_env
source aac_env/bin/activate
pip install -r requirements.txt

# Configure environment
cp src/aac/config/.env.example .env
# Edit .env with your API keys and settings

# Start services
python src/aac/models/enhanced_monitoring.py --start-dashboard &
python src/aac/agents/az_streamlit_interface.py &
```

## Cost Estimation

- **EC2 t3.micro**: $0-15/month (free tier)
- **S3 Storage**: $0.023/GB/month
- **Data Transfer**: $0-5/month
- **Total**: Under $20/month for basic setup

## Security Notes

- Use IAM roles instead of access keys when possible
- Enable MFA on your AWS account
- Regularly rotate access keys
- Monitor CloudWatch logs for security events
- Use KMS for encrypting sensitive configuration data

## Troubleshooting

- **Credentials Error**: Ensure AWS credentials are configured correctly
- **Region Error**: Check AWS_DEFAULT_REGION in .env
- **Permission Error**: Ensure IAM user has EC2, S3, and CloudWatch permissions
- **Instance Not Starting**: Check AWS console for error messages

## Advanced Usage

### Custom Deployment
```python
from src.aac.integrations.aws_integration import AWSIntegration

aws = AWSIntegration()
await aws.deploy_to_ec2(instance_type='t3.small')
```

### Direct AWS Operations
```python
# Backup specific files
await aws.backup_to_s3('src/aac/config', 'backups/config')

# Encrypt sensitive data
encrypted = await aws.encrypt_sensitive_data('my-secret-key')

# Setup monitoring
await aws.setup_monitoring('i-1234567890abcdef0')
```