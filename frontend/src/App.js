import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub, faLinkedin } from "@fortawesome/free-brands-svg-icons";
import './App.css';
import logo from "./nnv_logo.png"
import "./Spinner.css"


function App() {
    const [file, setFile] = useState(null);
    const [error, setError] = useState('');
    const [hiddenLayers, setHiddenLayers] = useState(1);
    const [iterations, setIterations] = useState(5);
    const [activationFunction, setActivationFunction] = useState("none");
    const [trainingDataCount, setTrainingDataCount] = useState(50);
    const [imageNumber, setImageNumber] = useState("None");
    const [mnistImage, setMnistImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [videoUrl, setVideoUrl] = useState(null);
    const [visualizationState, setVisualizationState] = useState('initial');

    const API_URL = process.env.REACT_APP_API_URL || 'https://neural-network-visualizer-284688458200.us-central1.run.app';

    useEffect(() => {
        if (imageNumber !== "None" && imageNumber !== "") {
            fetchMnistImage();
        } else {
            setMnistImage(null);
        }
    }, [imageNumber]);

    const fetchMnistImage = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${API_URL}/get_mnist_image/${imageNumber}`);

            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error response body:', errorText);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const contentType = response.headers.get('content-type');
            console.log('Content type:', contentType);

            if (!contentType || !contentType.includes('image/png')) {
                throw new Error(`Unexpected content type: ${contentType}`);
            }

            const imageBlob = await response.blob();
            const imageObjectURL = URL.createObjectURL(imageBlob);
            setMnistImage(imageObjectURL);
            setLoading(false);
        } catch (error) {
            console.error('Detailed error fetching MNIST image:', error);
            setError(`Failed to load MNIST image: ${error.message}`);
            setLoading(false);
        }
    };

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setError('');
        }
    };

    const handleHiddenLayersChange = (event) => {
        setHiddenLayers(event.target.value);
    };

    const handleIterationsChange = (event) => {
        setIterations(event.target.value);
    };

    const handleActivationChange = (event) => {
        setActivationFunction(event.target.value);
    };

    const handleTrainingDataChange = (event) => {
        setTrainingDataCount(event.target.value);
    }

    const handleImageNumberChange = (event) => {
        const value = event.target.value;
        if (value === "") {
            setImageNumber("None");
        } else {
            setImageNumber(value);
        }
    }

    const handleSubmit = async () => {
        if (activationFunction === "none") {
            console.log("No activation function");
            return;
        }

        setVisualizationState('training');
        setVideoUrl(null);

        const formData = new FormData();
        formData.append("hidden_layers", hiddenLayers);
        formData.append("iterations", iterations);
        formData.append("activation_function", activationFunction);
        formData.append("training_data_count", trainingDataCount);
        formData.append("image_number", imageNumber);

        if (file) {
            formData.append("file", file);
        }

        try {
            const response = await fetch(`${API_URL}/process`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                setVisualizationState('error');
                return;
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("image/gif")) {
                const videoBlob = await response.blob();
                const videoObjectURL = URL.createObjectURL(videoBlob);
                setVideoUrl(videoObjectURL);

                setVisualizationState('completed');
            } else {
                const result = await response.json();
                console.log(result);
                setVisualizationState('error');
            }
        } catch (error) {
            console.error("Error processing request:", error);
            setVisualizationState('error');
        }
    };

    const renderVisualizationArea = () => {
        switch(visualizationState) {
            case 'initial':
                return (
                    <div className="visualization-placeholder">
                        <p>Select your parameters to generate visualization</p>
                    </div>
                );
            case 'training':
                return (
                    <div className="visualization-loading">
                        <p>Training neural network model...</p>
                        <div className="spinner-container">
                            <div className="spinner"></div>
                        </div>
                    </div>
                );
            case 'completed':
                return videoUrl ? (
                    <div style={{ width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <img
                            src={videoUrl}
                            alt="Neural Network Visualization"
                            style={{
                                maxWidth: '100%',
                                maxHeight: '95%',
                                objectFit: 'contain'
                            }}
                            onLoad={() => console.log('Image loaded successfully')}
                            onError={(e) => {
                                console.error('Image loading error', e);
                                console.error('Image source:', videoUrl);
                            }}
                        />
                    </div>
                ) : null;
            case 'error':
                return (
                    <div className="visualization-error">
                        <p>Error generating visualization. Please check your parameters.</p>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div>
            <!-- Google tag (gtag.js) -->
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-0PN7PG4DSJ"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments)};
                gtag('js', new Date());

                gtag('config', 'G-0PN7PG4DSJ');
            </script>
            <div className="Header">
                <div className="Title">
                    <img src={logo} alt="logo" height="90%"></img>
                    <div className="NNV">NEURAL NETWORK VISUALIZER</div>
                </div>
                <div className="Contact">
                    <a href="https://github.com/CrestyDY/nnv" target="_blank" rel="noopener noreferrer">
                        <FontAwesomeIcon icon={faGithub} size="2x" style={{ color: "#00b3b3" }} />
                    </a>
                    <a href="https://www.linkedin.com/in/daniel-ti-yu/" target="_blank" rel="noopener noreferrer">
                        <FontAwesomeIcon icon={faLinkedin} size="2x" style={{ color: "#00b3b3" }} />
                    </a>
                </div>
            </div>

            <div className="main-container">
                <aside className="sidebar">
                    <div className="slider-container">
                        <label className="slider-label">Hidden Layers: {hiddenLayers}</label>
                        <input type="range" min="1" max="6" value={hiddenLayers} onChange={handleHiddenLayersChange}
                               className="slider"/>
                    </div>

                    <div className="slider-container">
                        <label className="slider-label">Iterations: {iterations}</label>
                        <input type="range" min="5" max="100" value={iterations} onChange={handleIterationsChange}
                               className="slider"/>
                    </div>

                    <div className="slider-container">
                        <label className="slider-label">Training Data: {trainingDataCount}</label>
                        <input type="range" min="50" max="1500" value={trainingDataCount}
                               onChange={handleTrainingDataChange}
                               className="slider"/>
                    </div>

                    <div className="dropdown-container">
                        <label className="dropdown-label">Select your activation function</label>
                        <div className="dropdown-content">
                            <select className="activation-dropdown" value={activationFunction}
                                    onChange={handleActivationChange}>
                                <option value="none">None selected</option>
                                <option value="tanh">TanH</option>
                                <option value="sigmoid">Sigmoid</option>
                                <option value="softmax">Softmax</option>
                                <option value="relu">ReLu</option>
                            </select>
                        </div>
                    </div>

                    <div className="input-container">
                        <label className="input-label">Image Number: {imageNumber}</label>
                        <input
                            type="number"
                            id="imageNumber"
                            name="imageNumber"
                            min="1"
                            max="60000"
                            value={imageNumber === "None" ? "" : imageNumber}
                            onChange={handleImageNumberChange}
                            placeholder="Enter image number"
                        />
                    </div>
                    <div className="mnist">
                        {loading ? (
                            <div className="loading-container">
                                <p>Loading MNIST image...</p>
                            </div>
                        ) : mnistImage ? (
                            <div className="mnist-display">
                                <img
                                    src={mnistImage}
                                    alt={`MNIST digit ${imageNumber}`}
                                    className="mnist-image"
                                />
                                <p className="mnist-label">MNIST Number: {imageNumber}</p>
                            </div>
                        ) : (
                            <div className="placeholder-text">
                                <p>Select an image number to display the corresponding MNIST image</p>
                            </div>
                        )}
                    </div>

                    <button className="submit-button" onClick={handleSubmit}>
                        GENERATE VISUALIZATION
                    </button>
                </aside>

                <main className="visualization-area">
                    {renderVisualizationArea()}
                </main>
            </div>
        </div>
    );
}

export default App;