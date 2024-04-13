# Python
FROM python:3.10

# Set the working directory
WORKDIR /app
COPY . ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the entrypoint command for running your application
CMD ["gunicorn", "main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "0", "--bind", "0.0.0.0:8000"]

