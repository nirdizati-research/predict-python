#!/bin/bash
eval "$(ssh-agent -s)"
chmod 600 ./cloud.key
echo -e "Host ${SERVER_IP}\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
ssh-add ./cloud.key
ssh -i ./cloud.key ${SERVER_USER}@${SERVER_IP} pwd

