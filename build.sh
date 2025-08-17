#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

python website/manage.py collectstatic --no-input

python website/manage.py migrate
