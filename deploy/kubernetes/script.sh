#!/bin/bash

# Kubernetes deployment script for AI Agent MS
# Converts Docker run commands to Kubernetes deployments

echo "🚀 Deploying AI Agent MS to Kubernetes..."

# Ensure minikube is running
echo "📋 Checking minikube status..."
if ! minikube status > /dev/null 2>&1; then
    echo "⚠️  Minikube is not running. Starting minikube..."
    minikube start
fi

# Load Docker images into minikube (if using local images)
echo "📦 Loading Docker images into minikube..."
minikube image load hdthinh1012/ai-agent-ms:latest
minikube image load hdthinh1012/ai-agent-ms-nginx:latest

# Create deployment with imagePullPolicy set to Never to use local images
echo "🚀 Creating deployment with local images..."
kubectl run ai-agent-ms --image=hdthinh1012/ai-agent-ms:latest --port=5000 --image-pull-policy=Never

# Alternative: Create a proper deployment file
echo "📝 Creating deployment using YAML configuration..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-ms
  labels:
    app: ai-agent-ms
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-agent-ms
  template:
    metadata:
      labels:
        app: ai-agent-ms
    spec:
      containers:
      - name: ai-agent-ms
        image: hdthinh1012/ai-agent-ms:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: ai-agent-ms-service
spec:
  selector:
    app: ai-agent-ms
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
EOF
