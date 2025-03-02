import React, { useState } from 'react';
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub, faLinkedin } from "@fortawesome/free-brands-svg-icons";
import { faUpload } from "@fortawesome/free-solid-svg-icons";
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            // Add file validation here if needed
            setFile(selectedFile);
            setError('');
        }
    };

    return (
        <div>
            {/* Keep your existing header */}
            <div className="Header">
                <div className="Title">
                    <div className="NNV">
                        Neural Network Visualiser
                    </div>
                </div>
                <div className="Contact">
                    <a href="https://github.com/" target="_blank" rel="noopener noreferrer">
                        <FontAwesomeIcon icon={faGithub} size="2x" style={{ color: '#00b3b3' }} />
                    </a>
                    <a href="https://linkedin.com/" target="_blank" rel="noopener noreferrer">
                        <FontAwesomeIcon icon={faLinkedin} size="2x" style={{ color: '#00b3b3' }} />
                    </a>
                </div>
            </div>

            {/* New main content area */}
            <div className="main-container">
                <aside className="sidebar">
                    <div className="upload-section">
                        <h2 className="upload-title">Upload Network</h2>
                        <label className="upload-area">
                            <FontAwesomeIcon icon={faUpload} size="2x" className="upload-icon"/>
                            <p className="upload-text">Click to upload or drag and drop</p>
                            <p className="upload-text">Neural network files</p>
                            <input
                                type="file"
                                onChange={handleFileChange}
                                accept=".json,.h5,.pkl"
                                style={{display: 'none'}}
                            />
                        </label>

                        {error && (
                            <div className="error-message">{error}</div>
                        )}

                        {file && (
                            <div className="file-info">
                                Selected: {file.name}
                            </div>
                        )}
                    </div>
                    <div className="slider-container">
                        <label className="slider-label">Hidden Layers</label>
                        <input type="range" min="1" max="5" className="slider"/>
                    </div>
                    <div className="slider-container">
                        <label className="slider-label">Max Inputs</label>
                        <input type="range" min="10" max="100" className="slider"/>
                    </div>
                    <div className="slider-container">
                        <label className="slider-label">Iterations</label>
                        <input type="range" min="5" max="100" className="slider"/>
                    </div>
                    <div className="dropdown-container">
                        <label className="dropdown-label">Select your activation function</label>
                        <div className="dropdown-content">
                            <select className="activation-dropdown">
                                <option value="none">None selected</option>
                                <option value="tanh">TanH</option>
                                <option value="sigmoid">Sigmoid</option>
                                <option value="softmax">Softmax</option>
                            </select>
                        </div>
                    </div>
                </aside>
                <main className="visualization-area">
                    {/* Your visualization content will go here */}
                </main>
            </div>
        </div>
    );
}

export default App;