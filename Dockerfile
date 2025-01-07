FROM ghcr.io/prefix-dev/pixi:0.39.3-jammy-cuda-12.2.2

# Copy pixi.toml and lock file
COPY pixi.toml pixi.lock ./
COPY . .

# Install dependencies
RUN pixi install

# Make process_data.py executable
RUN chmod +x process_data.py

# Executable container
ENTRYPOINT ["pixi", "run", "python", "./process_data.py"]

# Default input directory is /data
CMD ["/data"]
