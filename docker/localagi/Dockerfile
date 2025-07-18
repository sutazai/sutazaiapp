FROM golang:1.21-alpine AS builder

WORKDIR /app
RUN apk add --no-cache git
RUN git clone https://github.com/mudler/LocalAGI.git .
RUN go mod download
RUN go build -o localagi .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/localagi .
EXPOSE 8080
CMD ["./localagi"]
