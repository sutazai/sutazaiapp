# SutazAI Web UI

This directory contains the front-end components for the SutazAI web interface.

## Directory Structure

- `assets/`: Static assets such as images, fonts, and other media files
- `components/`: Reusable UI components
- `public/`: Publicly accessible files (favicon, index.html, etc.)
- `src/`: Source code for the web application

## Development

To start development:

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

## Building for Production

To build the application for production:

```
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## Integration with Backend

The web UI communicates with the SutazAI backend through API endpoints defined in the `backend` directory. See the API documentation for more details.
