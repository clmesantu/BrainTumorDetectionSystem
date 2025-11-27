const express = require('express');
const path = require('path');
const { exec } = require('child_process');

const app = express();
app.use(express.json());

// Serve the HTML file for the root URL
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'app.html'));
});

// Endpoint to train the model
app.post('/train-model', (req, res) => {
    exec('python train_and_save_model.py', (error, stdout, stderr) => {
        if (error) {
            return res.status(500).json({ status: 'error', message: error.message });
        }
        res.json({ status: 'success', message: stdout });
    });
});

// Endpoint to run the app
app.post('/run-app', (req, res) => {
    exec('python app.py', (error) => {
        if (error) {
            return res.status(500).json({ status: 'error', message: error.message });
        }
    });
    res.json({ status: 'success', message: 'App is running! Access it at http://127.0.0.1:5000' });
});

// Serve static files (if needed)
app.use(express.static(__dirname));

// Start the server
app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});
