# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /lib/apt/lists/*

# Set up a new user 'user' with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose the Hugging Face Spaces default port
EXPOSE 7860

# The command to run the environment server
# 0.0.0.0 allows external access (essential for HF)
# 7860 is the port HF monitors
CMD ["python", "-m", "openenv", "run", "--host", "0.0.0.0", "--port", "7860"]
