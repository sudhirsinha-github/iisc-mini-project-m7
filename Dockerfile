# Pull Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .  
COPY matrix_multiplication_MPI-0.0.1-py3-none-any.whl /app/

# Copy application files
COPY matrix_multiplication_MPI /app/matrix_multiplication_MPI

# Update pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r  matrix_multiplication_MPI/requirements.txt

# Expose port for application
EXPOSE 8001

# Start FastAPI application
CMD ["python", "matrix_multiplication_MPI/app/main.py"]
