# Use a minimal Python 3.10 image for efficiency
FROM python:3.10

# Optional: Set working directory (adjust if needed)
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-dev libx11-dev libxrender-dev



# Copy your Streamlit application code
COPY . .

# Optional: Install streamlit_login_auth_ui if not in requirements.txt
RUN pip3 install streamlit_login_auth_ui  # Uncomment if needed

# Command to run your Streamlit app (replace with your actual command)
CMD ["streamlit","run", "app.py","--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false", "--server.port", "8080"]
