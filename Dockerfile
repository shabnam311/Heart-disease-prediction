# Use the official lightweight Python image.
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Run the web service on container startup using gunicorn.
# The HF Space runs on port 7860 by default.
CMD exec gunicorn --bind :7860 --workers 1 --threads 8 --timeout 0 app:app
