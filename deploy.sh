#!/bin/bash
set -o nounset
set -o errexit

eval "$(ssh-agent -s)"
chmod 600 ./cloud.key
echo -e "Host ${SERVER_IP}\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
ssh-add ./cloud.key

# Open SSH tunnel to forward remote Docker socket to local docker.sock file
# SSH is put in control-socket mode, to close the connection when we have finished
ssh -M -S my-ctrl-socket -fnNT -o ExitOnForwardFailure=yes -L /tmp/docker.sock:/var/run/docker.sock ${SERVER_USER}@${SERVER_IP}

# Tell docker-compose to use the remote socket to talk to the DOcker daemon on the server
export DOCKER_HOST=unix:///tmp/docker.sock

docker-compose pull

# Start up the new containers
docker-compose up --detach --force-recreate server scheduler worker

# Close the SSH connection using the control socket opened previously
ssh -S my-ctrl-socket -O exit ${SERVER_USER}@${SERVER_IP}
