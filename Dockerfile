# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install Python 3.8 (or the version you prefer)
RUN apt-get update && apt-get install -y python3-pip python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Set up a working directory
WORKDIR /usr/src/app

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Command to run your application
CMD ["/usr/bin/python", "test.py"]