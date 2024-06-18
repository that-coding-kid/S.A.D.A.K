# Use a minimal Python 3.10 image for efficiency
FROM python:3.10-slim

# Optional: Set working directory (adjust if needed)
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy your Streamlit application code
COPY . .

# Optional: Install streamlit_login_auth_ui if not in requirements.txt
RUN pip3 install streamlit_login_auth_ui  # Uncomment if needed

# Command to run your Streamlit app (replace with your actual command)
CMD ["python3", "app.py"]
