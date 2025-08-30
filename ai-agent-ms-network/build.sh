docker build -t $(minikube ip):53507/hdthinh1012/ai-agent-ms:latest ai-agent-ms/
docker build -t $(minikube ip):53507/hdthinh1012/ai-agent-ms-nginx:latest ai-agent-ms-nginx/

docker push $(minikube ip):53507/hdthinh1012/ai-agent-ms:latest
docker push $(minikube ip):53507/hdthinh1012/ai-agent-ms-nginx:latest