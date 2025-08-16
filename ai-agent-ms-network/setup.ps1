# setup network
docker network create ai-agent-ms-network
# run ai-agent-ms
docker run --env-file ai-agent-ms/.env -d --name ai-agent-ms --network ai-agent-ms-network -p 5000:5000 hdthinh1012/ai-agent-ms:latest
# run ai-agent-ms-nginx
docker run -d --name ai-agent-ms-nginx --network ai-agent-ms-network -p 80:80 hdthinh1012/ai-agent-ms-nginx:latest