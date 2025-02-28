import React, { useState } from 'react';
import axios from 'axios';
import { 
    Box, 
    Button, 
    Typography, 
    LinearProgress, 
    Alert,
    Grid,
    Paper
} from '@mui/material';

const DiagramUploader = () => {
    const [file, setFile] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStatus, setUploadStatus] = useState({
        success: false,
        error: false,
        message: ''
    });
    const [analyzedResult, setAnalyzedResult] = useState(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        
        // Validate image type
        const allowedTypes = [
            'image/png', 
            'image/jpeg', 
            'image/bmp', 
            'image/tiff'
        ];
        
        if (selectedFile && allowedTypes.includes(selectedFile.type)) {
            setFile(selectedFile);
            setUploadStatus({ success: false, error: false, message: '' });
        } else {
            setUploadStatus({
                success: false,
                error: true,
                message: 'Invalid image type. Please upload PNG, JPEG, BMP, or TIFF.'
            });
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setUploadStatus({
                success: false,
                error: true,
                message: 'Please select an image first.'
            });
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            setUploadProgress(0);
            setUploadStatus({ success: false, error: false, message: '' });

            const response = await axios.post('/doc/diagram/analyze', formData, {
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

            setAnalyzedResult(result);
            setUploadStatus({
                success: true,
                error: false,
                message: 'Diagram uploaded and analyzed successfully!'
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
        <Box sx={{ maxWidth: 600, margin: 'auto', padding: 2 }}>
            <Typography variant="h6" gutterBottom>
                Diagram Analyzer
            </Typography>

            <Grid container spacing={2} alignItems="center">
                <Grid item xs={8}>
                    <input
                        type="file"
                        accept=".png,.jpg,.jpeg,.bmp,.tiff"
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                        id="diagram-upload-input"
                    />
                    <label htmlFor="diagram-upload-input">
                        <Button 
                            variant="contained" 
                            component="span"
                            color={file ? 'success' : 'primary'}
                            fullWidth
                        >
                            {file ? file.name : 'Select Diagram'}
                        </Button>
                    </label>
                </Grid>
                <Grid item xs={4}>
                    <Button 
                        variant="contained" 
                        onClick={handleUpload} 
                        disabled={!file}
                        fullWidth
                    >
                        Analyze
                    </Button>
                </Grid>
            </Grid>

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

            {analyzedResult && (
                <Paper 
                    elevation={3} 
                    sx={{ 
                        mt: 2, 
                        p: 2, 
                        backgroundColor: 'background.default' 
                    }}
                >
                    <Typography variant="h6" gutterBottom>
                        Diagram Analysis Results
                    </Typography>
                    <Box sx={{ 
                        maxHeight: 400, 
                        overflowY: 'auto',
                        backgroundColor: 'background.paper',
                        p: 1,
                        borderRadius: 1
                    }}>
                        <pre>{JSON.stringify(analyzedResult, null, 2)}</pre>
                    </Box>
                </Paper>
            )}
        </Box>
    );
};

export default DiagramUploader; 