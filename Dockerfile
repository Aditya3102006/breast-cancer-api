# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for compiling certain packages or system tasks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face sentence-transformer embedding model 
# so it is cached in the docker image for faster startup times.
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# Copy the rest of the application files into the container
COPY . .

# Expose the port Streamlit or FastAPI will run on
EXPOSE 7860

# Command to run the application
# We use port 7860 as it is standard for Hugging Face Spaces Docker SDK.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
