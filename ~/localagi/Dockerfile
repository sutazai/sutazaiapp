FROM python:3.10-bullseye
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Remove audio dependencies
ENV DEBIAN_FRONTEND noninteractive

COPY . /app
RUN pip install .
ENTRYPOINT [ "python", "./main.py", "--skip-avatar" ]; 