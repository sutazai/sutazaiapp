FROM node:18-alpine

WORKDIR /app

# Install git
RUN apk add --no-cache git

# Clone BigAGI
RUN git clone https://github.com/enricoros/big-AGI.git .

# Install dependencies
RUN npm install

# Build the application
RUN npm run build

# Create startup script
RUN echo '#!/bin/sh\n\
npm start\n\
' > start.sh && chmod +x start.sh

EXPOSE 3000


# Run as non-root user
USER nobody
CMD ["./start.sh"]
