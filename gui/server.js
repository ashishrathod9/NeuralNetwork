const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static('public'));

// Serve the GUI
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Endpoint to train the neural network
app.post('/train', (req, res) => {
    const { epochs, learningRate, hiddenSize, dataset } = req.body;
    
    // Compile and run the C++ training program
    const trainingProcess = spawn('make', ['-C', '../build'], { cwd: path.join(__dirname) });
    
    let output = '';
    
    trainingProcess.stdout.on('data', (data) => {
        output += data.toString();
    });
    
    trainingProcess.stderr.on('data', (data) => {
        output += data.toString();
    });
    
    trainingProcess.on('close', (code) => {
        if (code === 0) {
            res.json({ 
                success: true, 
                message: 'Training completed successfully',
                output: output,
                epochs: epochs,
                finalEpoch: epochs
            });
        } else {
            res.json({ 
                success: false, 
                message: 'Training failed',
                error: output
            });
        }
    });
});

// Endpoint to get network status
app.get('/status', (req, res) => {
    res.json({ 
        status: 'ready',
        available_datasets: ['XOR', 'AND', 'OR', 'NOT'],
        last_training: new Date().toISOString()
    });
});

app.listen(port, () => {
    console.log(`Neural Network GUI server running at http://localhost:${port}`);
});