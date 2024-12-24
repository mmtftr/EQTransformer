FROM ghcr.io/prefix-dev/pixi:0.39.3-jammy-cuda-12.2.2

# Copy pixi.toml and lock file
COPY pixi.toml pixi.lock* ./

# Install dependencies
RUN pixi install


# Copy source code
COPY create_stations_json.py .
COPY organize_gcf_files.py .
COPY preprocess_gcf.py .
COPY process_data.py .

# Make process_data.py executable
RUN chmod +x process_data.py

# Set the entrypoint to use pixi shell
ENTRYPOINT ["pixi", "shell", "--"]

# Set the default command to run process_data.py
CMD ["./process_data.py", "/data"]