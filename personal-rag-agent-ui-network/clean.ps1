# stop and remove personal-rag-agent-ui
docker stop personal-rag-agent-ui
docker rm personal-rag-agent-ui
# stop and remove personal-rag-agent-ui-nginx-sidecar
docker stop personal-rag-agent-ui-nginx-sidecar
docker rm personal-rag-agent-ui-nginx-sidecar
# remove personal-rag-agent-ui-network
docker network rm personal-rag-agent-ui-network
