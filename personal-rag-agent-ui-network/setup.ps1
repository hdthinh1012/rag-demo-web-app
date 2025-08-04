# setup network
docker network create personal-rag-agent-ui-network
# run personal-rag-agent-ui
docker run -d --name personal-rag-agent-ui --network personal-rag-agent-ui-network -p 3000:3000 hdthinh1012/personal-rag-agent-ui:latest