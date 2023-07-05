#!/bin/bash

# Checking if a string is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <package to install environment as>"
    exit 1
fi

NEW_NAME="$1"
MOD_NAME="${NEW_NAME//-/_}"
sed -i "s/gym_env/$MOD_NAME/g" DQN_DogFight.py
sed -i "s/gym_env/$MOD_NAME/g" ./gym-env/setup.py
sed -i "s/gym_env/$MOD_NAME/g" ./gym-env/gym_env/__init__.py
sed -i "s/gym_env/$MOD_NAME/g" ./gym-env/gym_env/envs/__init__.py
sed -i "s/gym_env/$MOD_NAME/g" ./gym-env/gym_env/envs/__init__.py

mv ./gym-env/gym_env "$MOD_NAME"
mv ./gym-env "$NEW_NAME"