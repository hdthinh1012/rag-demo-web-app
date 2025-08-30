# Use continuumio/miniconda3 base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment_docker.yml /app/environment.yml

# Create conda environment

RUN conda env create -f environment.yml

RUN conda activate rag-demo

RUN CONDA_VERBOSITY=3 conda init 

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "rag-demo", "/bin/bash", "-c"]

# Copy application files
COPY . /app/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Activate conda environment and run the application
CMD ["conda", "run", "--no-capture-output", "-n", "rag-demo", "python", "app.py"]
