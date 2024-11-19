FROM python:3.9-slim

# Set working directory
WORKDIR /usr/src/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port 8080
EXPOSE 8080

# Set environment variable for Cloud Run
ENV PORT=8080

# Command to start your app
CMD ["gunicorn", "-b", ":8080", "app:app"]
