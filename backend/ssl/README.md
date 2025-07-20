# SSL Certificates

This directory is used to store SSL certificates for the backend.

## Generating a new self-signed certificate

To generate a new self-signed certificate, run the following command:

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```
