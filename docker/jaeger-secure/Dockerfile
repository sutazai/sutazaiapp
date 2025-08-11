FROM jaegertracing/all-in-one:latest

# Switch to existing jaeger user (it already exists in the base image)
USER 10001

# Jaeger ports
EXPOSE 16686 14268 14250 9411 4317 4318

# Use default entrypoint from base image
ENTRYPOINT ["/go/bin/all-in-one-linux"]