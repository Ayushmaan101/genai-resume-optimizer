# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# The command to run your app, with telemetry disabled
CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--browser.gatherUsageStats", "false"]