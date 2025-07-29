label-studio start
Runs on: http://localhost:8080





label-studio-ml start gpt_autolabel.py
This will start a backend API (default: http://localhost:9090).

Go to Settings → Machine Learning in your Label Studio project.

Add a new ML backend:

URL: http://localhost:9090

Enable Auto Labeling → It will call GPT to pre-fill labels.