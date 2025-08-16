# setup network
docker network create personal-rag-agent-ui-network
# run personal-rag-agent-ui
docker run -d --name personal-rag-agent-ui --network personal-rag-agent-ui-network -p 3000:3000 hdthinh1012/personal-rag-agent-ui:latest
# run personal-rag-agent-ui-nginx-sidecar
docker run -d --name personal-rag-agent-ui-nginx-sidecar --network personal-rag-agent-ui-network -p 444:444 -p 81:80 hdthinh1012/personal-rag-agent-ui-nginx-sidecar:latest