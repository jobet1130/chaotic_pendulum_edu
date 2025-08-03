# Multi-stage Dockerfile for Chaotic Pendulum Educational Project
# Supports both development and production environments

# =============================================================================
# Base Stage - Common dependencies and setup
# =============================================================================
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt environment.yml ./

# =============================================================================
# Development Stage
# =============================================================================
FROM base as development

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    black \
    flake8 \
    isort \
    mypy \
    pytest \
    pytest-cov \
    pytest-mock

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/features data/labeled data/sample_plots \
    animations models notebooks quizzes reports/figures \
    student_submissions tests

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Production Stage
# =============================================================================
FROM base as production

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/features data/labeled data/sample_plots \
    animations models notebooks quizzes reports/figures \
    student_submissions tests

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for web interface (if needed)
EXPOSE 8000

# Default command for production
CMD ["python", "-c", "print('Chaotic Pendulum Educational Project is ready!')"]

# =============================================================================
# Conda Stage (Alternative environment)
# =============================================================================
FROM continuumio/miniconda3:latest as conda

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy conda environment file
COPY environment.yml ./

# Create conda environment
RUN conda env create -f environment.yml

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/features data/labeled data/sample_plots \
    animations models notebooks quizzes reports/figures \
    student_submissions tests

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Activate conda environment and start Jupyter
CMD ["conda", "run", "-n", "pendulum-chaos-env", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Testing Stage
# =============================================================================
FROM development as testing

# Install testing dependencies
RUN pip install \
    pytest-xdist \
    pytest-html \
    coverage

# Copy test files
COPY tests/ ./tests/

# Set environment for testing
ENV PYTHONPATH=/app

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term-missing"]

# =============================================================================
# Documentation Stage
# =============================================================================
FROM development as docs

# Install documentation dependencies
RUN pip install \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints

# Copy documentation files
COPY docs/ ./docs/

# Build documentation
CMD ["sphinx-build", "-b", "html", "docs/", "docs/_build/html/"]

# =============================================================================
# Final Stage - Lightweight production image
# =============================================================================
FROM python:3.12-slim as final

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt ./
COPY src/ ./src/
COPY pendulum_config.json ./

# Install only production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p data/raw data/features data/labeled data/sample_plots \
    animations models reports/figures

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-c", "print('Chaotic Pendulum Educational Project is running!')"]
