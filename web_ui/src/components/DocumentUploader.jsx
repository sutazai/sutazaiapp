import React, { useState } from 'react';
import axios from 'axios';
import { 
    Box, 
    Button, 
    Typography, 
    LinearProgress, 
    Alert 
} from '@mui/material';

const DocumentUploader = () => {
    const [file, setFile] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState({
        success: false,
        error: false,
        message: ''
    });
    const [parsedResult, setParsedResult] = useState(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        
        // Validate file type
        const allowedTypes = [
            'application/pdf', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ];
        
        if (selectedFile && allowedTypes.includes(selectedFile.type)) {
            setFile(selectedFile);
            setUploadStatus({ success: false, error: false, message: '' });
        } else {
            setUploadStatus({
                success: false,
                error: true,
                message: 'Invalid file type. Please upload PDF or DOCX.'
            });
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setUploadStatus({
                success: false,
                error: true,
                message: 'Please select a file first.'
            });
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            setUploadProgress(0);
            setUploadStatus({ success: false, error: false, message: '' });

            const response = await axios.post('/doc/parse', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    setUploadProgress(percentCompleted);
                }
            });

            // Simulate result retrieval (replace with actual backend logic)
            const result = await new Promise(resolve => {
                setTimeout(() => {
                    resolve(response.data);
                }, 2000);
            });

            setParsedResult(result);
            setUploadStatus({
                success: true,
                error: false,
                message: 'Document uploaded and parsed successfully!'
            });
        } catch (error) {
            setUploadStatus({
                success: false,
                error: true,
                message: error.response?.data?.detail || 'Upload failed'
            });
        }
    };

    return (
        <Box sx={{ maxWidth: 500, margin: 'auto', padding: 2 }}>
            <Typography variant="h6" gutterBottom>
                Document Parser
            </Typography>

            <input
                type="file"
                accept=".pdf,.docx"
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="document-upload-input"
            />
            <label htmlFor="document-upload-input">
                <Button 
                    variant="contained" 
                    component="span"
                    color={file ? 'success' : 'primary'}
                >
                    {file ? file.name : 'Select Document'}
                </Button>
            </label>

            <Button 
                variant="contained" 
                onClick={handleUpload} 
                disabled={!file}
                sx={{ ml: 2 }}
            >
                Upload & Parse
            </Button>

            {uploadProgress > 0 && (
                <LinearProgress 
                    variant="determinate" 
                    value={uploadProgress} 
                    sx={{ mt: 2 }}
                />
            )}

            {uploadStatus.message && (
                <Alert 
                    severity={uploadStatus.error ? 'error' : 'success'}
                    sx={{ mt: 2 }}
                >
                    {uploadStatus.message}
                </Alert>
            )}

            {parsedResult && (
                <Box sx={{ mt: 2, p: 2, border: '1px solid #ddd', borderRadius: 2 }}>
                    <Typography variant="h6">Parsing Results</Typography>
                    <pre>{JSON.stringify(parsedResult, null, 2)}</pre>
                </Box>
            )}
        </Box>
    );
};

export default DocumentUploader; 