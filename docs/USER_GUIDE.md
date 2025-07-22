# SutazaiApp User Guide

This guide will help you get started with SutazaiApp and explore its key features.

## Getting Started

### Accessing the API

SutazaiApp provides a REST API for all functionality. Once the application is running, you can access the API documentation at:

```
http://localhost:8000/docs
```

This interactive documentation allows you to explore and test all available endpoints.

## Key Features

### Document Processing

SutazaiApp can extract text and metadata from various document types, including PDF and DOCX files.

#### Uploading Documents

To process a document:

1. Prepare your PDF or DOCX file
2. Use an HTTP client (like curl, Postman, or your application) to upload the file to the `/documents/process` endpoint
3. The API will return structured data including:
   - Document metadata (title, author, page count, etc.)
   - Extracted text content
   - Document ID for later reference

#### Example Document Upload

Using curl:

```bash
curl -X POST "http://localhost:8000/documents/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf" \
  -F "store_vectors=true"
```

### Vector Search

SutazaiApp enables semantic search across processed documents, allowing you to find information based on meaning rather than just keywords.

#### Searching Documents

To search across documents:

1. Formulate your search query
2. Send a POST request to the `/vector/search` endpoint
3. Review the returned results, which include:
   - Matching text chunks
   - Document information
   - Relevance scores

#### Example Search

```bash
curl -X POST "http://localhost:8000/vector/search" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence applications in healthcare",
    "limit": 5
  }'
```

### Code Generation

SutazaiApp can generate code based on natural language specifications and improve existing code.

#### Generating Code

To generate code:

1. Write a specification describing what the code should do
2. Specify the target programming language
3. Send a POST request to the `/code/generate` endpoint
4. Review the generated code and any identified issues

#### Example Code Generation

```bash
curl -X POST "http://localhost:8000/code/generate" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "spec_text": "Create a function that calculates the average of numbers in a list",
    "language": "python"
  }'
```

#### Improving Code

To improve existing code:

1. Prepare your code and identified issues
2. Send a POST request to the `/code/improve` endpoint
3. Review the improved code and fixed issues

#### Example Code Improvement

```bash
curl -X POST "http://localhost:8000/code/improve" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calculate_average(numbers):\n    sum = 0\n    for i in range(len(numbers)):\n        sum += numbers[i]\n    return sum / len(numbers)",
    "issues": ["Could be more Pythonic", "Doesn't handle empty lists"],
    "language": "python"
  }'
```

## Advanced Usage

### System Metrics

SutazaiApp provides system metrics to monitor performance:

```bash
curl -X GET "http://localhost:8000/metrics" \
  -H "accept: application/json"
```

### Processing Large Documents

For large documents:

1. Increase the timeout value in your HTTP client
2. Consider splitting very large documents into smaller chunks
3. Monitor the processing status using the returned document ID

### Bulk Processing

For processing multiple documents:

1. Process each document individually
2. Store the returned document IDs
3. Use the vector search to query across all documents

## Troubleshooting

### Common Issues

1. **Document Processing Fails**
   - Ensure the document is not corrupted
   - Check if the file format is supported
   - Verify the file size is within limits

2. **Search Returns No Results**
   - Ensure documents have been processed with `store_vectors=true`
   - Try reformulating your query
   - Check if the document content is relevant to your query

3. **Code Generation Issues**
   - Make the specification more detailed and clear
   - Verify the requested language is supported
   - For complex code, break down into smaller, focused requests

### Getting Help

If you encounter issues not covered in this guide:

1. Check the application logs in the `logs/` directory
2. Review the API documentation for endpoint details
3. Check the system metrics for performance bottlenecks
4. Contact system administrators for further assistance

## Best Practices

1. **Document Processing**
   - Use OCR only when necessary (for scanned documents)
   - Provide well-formatted, searchable PDFs when possible
   - Process related documents together for better search context

2. **Vector Search**
   - Be specific with search queries
   - Use natural language rather than keywords
   - Experiment with different phrasings if initial results are poor

3. **Code Generation**
   - Provide clear, detailed specifications
   - Include expected inputs and outputs
   - Specify edge cases and error handling requirements 