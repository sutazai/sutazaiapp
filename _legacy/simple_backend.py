#\!/usr/bin/env python3
import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to Python path
sys.path.insert(0, '/opt/sutazaiapp')

app = FastAPI(title='SutazAI v8 Production API', version='2.0.0')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/health')
async def health_check():
    return JSONResponse({
        'status': 'healthy',
        'version': 'SutazAI v8 (2.0.0)',
        'message': 'Production backend is operational'
    })

@app.get('/')
async def root():
    return JSONResponse({
        'message': 'SutazAI v8 Production API',
        'version': '2.0.0',
        'status': 'running'
    })

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
