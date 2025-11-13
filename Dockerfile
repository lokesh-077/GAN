# Use an official PyTorch CPU base image so torch/torchvision are available at build time
FROM pytorch/pytorch:1.13.1-cpu

WORKDIR /app

# Copy requirements then install to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose Streamlit port
EXPOSE 8501

ENV PYTHONUNBUFFERED=1
# Run streamlit in server mode
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
