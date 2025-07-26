# SutazaiApp API Reference

This document provides a complete reference to the SutazaiApp REST API.

## Base URL

All API endpoints are relative to the base URL:
```
http://localhost:8000
```

## Health Check

### GET /health

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2023-06-01T12:34:56.789Z",
  "version": "1.0.0",
  "server_time": 1685626496.789
}
```

## Document Processing

### POST /documents/process

Process a document (PDF or DOCX) and extract text.

**Parameters:**

- `file`: The document file to process (multipart/form-data)
- `store_vectors`: Whether to store document vectors for search (boolean, default: true)

**Response:**

```json
{
  "metadata": {
    "title": "Example Document",
    "author": "John Doe",
    "subject": "Example",
    "page_count": 5,
    "file_size_bytes": 123456,
    "processed_at": "2023-06-01T12:34:56.789Z"
  },
  "pages": [
    {
      "page_num": 1,
      "text": "This is the content of page 1..."
    },
    {
      "page_num": 2,
      "text": "This is the content of page 2..."
    }
  ],
  "full_text": "This is the content of page 1...\n\nThis is the content of page 2...",
  "image_count": 3,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "vectorized": true,
  "processing_time_ms": 1234
}
```

## Vector Search

### POST /vector/search

Search for similar document chunks using semantic search.

**Request Body:**

```json
{
  "query": "your search query",
  "limit": 5
}
```

**Response:**

```json
{
  "query": "your search query",
  "results": [
    {
      "text": "This is a matching text chunk...",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "chunk_index": 0,
      "score": 0.95,
      "metadata": {
        "title": "Example Document",
        "author": "John Doe"
      }
    },
    {
      "text": "Another matching text chunk...",
      "document_id": "550e8400-e29b-41d4-a716-446655440001",
      "chunk_index": 2,
      "score": 0.87,
      "metadata": {
        "title": "Another Document",
        "author": "Jane Smith"
      }
    }
  ],
  "count": 2,
  "search_time_ms": 45
}
```

## Code Generation

### POST /code/generate

Generate code based on a specification.

**Request Body:**

```json
{
  "spec_text": "Write a function that calculates fibonacci numbers",
  "language": "python"
}
```

**Response:**

```json
{
  "language": "python",
  "spec_text": "Write a function that calculates fibonacci numbers",
  "generated_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n",
  "issues": [
    "Pylint: C0103: Variable name 'n' doesn't conform to snake_case naming style"
  ],
  "generation_time_ms": 2345
}
```

### POST /code/improve

Improve existing code based on identified issues.

**Request Body:**

```json
{
  "code": "def fib(n):\n    if n <= 1:\n        return n\n    else:\n        return fib(n-1) + fib(n-2)\n",
  "issues": [
    "This implementation is inefficient for large n due to excessive recursion"
  ],
  "language": "python"
}
```

**Response:**

```json
{
  "original_code": "def fib(n):\n    if n <= 1:\n        return n\n    else:\n        return fib(n-1) + fib(n-2)\n",
  "improved_code": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    \n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n",
  "original_issues": [
    "This implementation is inefficient for large n due to excessive recursion"
  ],
  "remaining_issues": [],
  "issues_fixed": 1,
  "improvement_time_ms": 1678
}
```

## System Metrics

### GET /metrics

Get system performance metrics.

**Parameters:**

- `limit`: Number of historical metrics to return (integer, default: 10)

**Response:**

```json
{
  "metrics": [
    {
      "timestamp": "2023-06-01T12:30:00.000Z",
      "cpu_percent": 45.2,
      "memory_percent": 62.7,
      "disk_usage_percent": 38.5,
      "document_processing": {
        "success_count": 12,
        "error_count": 1,
        "avg_processing_time_ms": 856.3
      },
      "code_generation": {
        "success_count": 8,
        "error_count": 0,
        "avg_generation_time_ms": 2345.7
      },
      "vector_searches": {
        "count": 25,
        "avg_time_ms": 43.2
      }
    }
  ],
  "current": {
    "timestamp": "2023-06-01T12:34:56.789Z",
    "cpu_percent": 48.3,
    "memory_percent": 64.1,
    "disk_usage_percent": 38.5,
    "document_processing": {
      "success_count": 15,
      "error_count": 1,
      "avg_processing_time_ms": 843.7
    },
    "code_generation": {
      "success_count": 10,
      "error_count": 0,
      "avg_generation_time_ms": 2187.9
    },
    "vector_searches": {
      "count": 32,
      "avg_time_ms": 41.8
    }
  }
} 