kubectl run ai-agent-ms --image=docker.io/hdthinh1012/ai-agent-ms:latest --image-pull-policy=Never --port=5000
kubectl expose pod ai-agent-ms --type=LoadBalancer --port=5000
minikube service ai-agent-ms

kubectl run ai-agent-ms-nginx --image=docker.io/hdthinh1012/ai-agent-ms-nginx:latest --image-pull-policy=Never --port=443
kubectl expose pod ai-agent-ms-nginx --type=LoadBalancer --port=443
minikube service ai-agent-ms-nginx