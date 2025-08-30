kubectl delete -f ./ai-agent-ms-network/kubernetes/ai-agent-ms-pod.yml --ignore-not-found
kubectl delete -f ./ai-agent-ms-network/kubernetes/ai-agent-ms-service.yml --ignore-not-found
kubectl delete -f ./ai-agent-ms-network/kubernetes/ai-agent-ms-ingress.yml --ignore-not-found

kubectl apply -f ./ai-agent-ms-network/kubernetes/ai-agent-ms-pod.yml
kubectl apply -f ./ai-agent-ms-network/kubernetes/ai-agent-ms-service.yml
kubectl apply -f ./ai-agent-ms-network/kubernetes/ai-agent-ms-ingress.yml
