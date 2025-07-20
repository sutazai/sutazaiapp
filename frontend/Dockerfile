# Dockerfile for Streamlit Frontend

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . /app

# Command to run the application
CMD ["streamlit", "run", "intelligent_chat_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
