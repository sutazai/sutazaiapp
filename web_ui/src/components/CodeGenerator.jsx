import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    Box, 
    Typography, 
    TextField, 
    Button, 
    Select, 
    MenuItem, 
    FormControl, 
    InputLabel, 
    Alert, 
    Paper,
    Grid
} from '@mui/material';
import { styled } from '@mui/material/styles';

const CodeDisplay = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    backgroundColor: theme.palette.background.default,
    fontFamily: 'monospace',
    whiteSpace: 'pre-wrap',
    overflowX: 'auto',
    maxHeight: '400px',
    overflowY: 'auto'
}));

const CodeGenerator = () => {
    const [specification, setSpecification] = useState('');
    const [modelName, setModelName] = useState('deepseek-coder');
    const [language, setLanguage] = useState('python');
    const [otp, setOtp] = useState('');
    const [generatedCode, setGeneratedCode] = useState(null);
    const [securityWarnings, setSecurityWarnings] = useState([]);
    const [error, setError] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);

    useEffect(() => {
        // Fetch available models from backend
        const fetchModels = async () => {
            try {
                const response = await axios.get('/code/models');
                setAvailableModels(response.data.models);
            } catch (err) {
                console.error('Failed to fetch models', err);
                setAvailableModels(['deepseek-coder', 'gpt4all']);
            }
        };

        fetchModels();
    }, []);

    const handleGenerate = async () => {
        // Reset previous state
        setGeneratedCode(null);
        setSecurityWarnings([]);
        setError(null);

        try {
            const response = await axios.post('/code/generate', {
                specification,
                model_name: modelName,
                language,
                otp
            });

            if (response.data.success) {
                setGeneratedCode(response.data.generated_code);
                setSecurityWarnings(response.data.security_warnings || []);
            } else {
                setError(response.data.error || 'Code generation failed');
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'An unexpected error occurred');
        }
    };

    return (
        <Box sx={{ maxWidth: 800, margin: 'auto', padding: 2 }}>
            <Typography variant="h4" gutterBottom>
                AI Code Generator
            </Typography>

            <Grid container spacing={2}>
                <Grid item xs={12} md={8}>
                    <TextField
                        fullWidth
                        multiline
                        rows={4}
                        variant="outlined"
                        label="Code Specification"
                        value={specification}
                        onChange={(e) => setSpecification(e.target.value)}
                        placeholder="Describe the code you want to generate..."
                    />
                </Grid>
                <Grid item xs={12} md={4}>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <FormControl fullWidth>
                                <InputLabel>Model</InputLabel>
                                <Select
                                    value={modelName}
                                    label="Model"
                                    onChange={(e) => setModelName(e.target.value)}
                                >
                                    {availableModels.map(model => (
                                        <MenuItem key={model} value={model}>
                                            {model}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                            <FormControl fullWidth>
                                <InputLabel>Language</InputLabel>
                                <Select
                                    value={language}
                                    label="Language"
                                    onChange={(e) => setLanguage(e.target.value)}
                                >
                                    {['python', 'javascript', 'rust', 'go'].map(lang => (
                                        <MenuItem key={lang} value={lang}>
                                            {lang}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="OTP"
                                type="password"
                                value={otp}
                                onChange={(e) => setOtp(e.target.value)}
                                placeholder="Enter OTP"
                            />
                        </Grid>
                    </Grid>
                </Grid>
            </Grid>

            <Button 
                variant="contained" 
                color="primary" 
                onClick={handleGenerate}
                sx={{ mt: 2 }}
                disabled={!specification || !otp}
            >
                Generate Code
            </Button>

            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                </Alert>
            )}

            {securityWarnings.length > 0 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Security Warnings:</Typography>
                    {securityWarnings.map((warning, index) => (
                        <Typography key={index} variant="body2">
                            - {warning.message || JSON.stringify(warning)}
                        </Typography>
                    ))}
                </Alert>
            )}

            {generatedCode && (
                <Box sx={{ mt: 2 }}>
                    <Typography variant="h6">Generated Code:</Typography>
                    <CodeDisplay>
                        {generatedCode}
                    </CodeDisplay>
                </Box>
            )}
        </Box>
    );
};

export default CodeGenerator; 