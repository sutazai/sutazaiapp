# Dockerfile for Semgrep
# ------------------------

# Use the official Semgrep image
FROM returntocorp/semgrep

# Set the working directory
WORKDIR /src

# Set the entrypoint
ENTRYPOINT ["semgrep"]
