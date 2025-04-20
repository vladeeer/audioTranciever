#!/bin/bash

VENV_NAME="venv"

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment in the ./$VENV_NAME directory..."
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment ./$VENV_NAME already exists."
fi

if [ -f "$VENV_NAME/bin/activate" ]; then
    echo "Activating the virtual environment..."
    source "$VENV_NAME/bin/activate"
else
    echo "Unable to activate the virtual environment. It does not exist."
    exit 1
fi