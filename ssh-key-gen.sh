#!/bin/bash

# Generate a new SSH key and add it to your machine's SSH agent
ssh-keygen -t ed25519 -C "your email"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

cat ~/.ssh/id_ed25519.pub

# Adding a new SSH key to your account
