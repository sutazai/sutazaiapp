# Dockerfile for SutazAI Streamlit Frontend
# -------------------------------------------

# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Copy the application code
COPY ./frontend /app/frontend
COPY ./frontend/enhanced_streamlit_app.py /app/main.py

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py"]
