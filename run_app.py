import subprocess
import sys
import time
import os

print("Starting Traffic Insight Pro Architecture...")

# Path configurations
cwd = os.path.dirname(os.path.abspath(__file__))

# 1. Start Flask API
print(">> Initializing Flask Backend API on Port 5000...")
flask_process = subprocess.Popen([sys.executable, "app.py"], cwd=cwd)

# Give Flask a second to boot up
time.sleep(2)

# 2. Start Streamlit Dashboard
print(">> Engaging Streamlit UI on Port 8501...")
streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py"], cwd=cwd)

try:
    print("\nSystem Online! Press Ctrl+C to terminate both servers.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down systems...")
    flask_process.terminate()
    streamlit_process.terminate()
    flask_process.wait()
    streamlit_process.wait()
    print("Goodbye!")
