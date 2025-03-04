import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub, faLinkedin } from "@fortawesome/free-brands-svg-icons";
import { faUpload } from "@fortawesome/free-solid-svg-icons";
import './App.css';
import logo from "./nnv_logo.png"

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

    // Fetch MNIST image whenever imageNumber changes
    useEffect(() => {
        // Only fetch if imageNumber is a valid number
        if (imageNumber !== "None" && imageNumber !== "") {
            fetchMnistImage();
        } else {
            // Clear the image if imageNumber is None or empty
            setMnistImage(null);
        }
    }, [imageNumber]);

    const fetchMnistImage = async () => {
        try {
            setLoading(true);
            const response = await fetch(`http://localhost:5000/get_mnist_image/${imageNumber}`);

            if (!response.ok) {
                throw new Error('Failed to fetch image');
            }

            const imageBlob = await response.blob();
            const imageObjectURL = URL.createObjectURL(imageBlob);
            setMnistImage(imageObjectURL);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching MNIST image:', error);
            setError('Failed to load MNIST image');
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
        // If the input is empty, set it back to "None"
        if (value === "") {
            setImageNumber("None");
        } else {
            setImageNumber(value);
        }
    }

    const handleSubmit = async () => {
        console.log(hiddenLayers);
        console.log(imageNumber);
        console.log(activationFunction);
        console.log(trainingDataCount);
        console.log(iterations);
        const formData = new FormData();
        if (file) {
            formData.append("file", file);
        }
        if (activationFunction === "none") {
            console.log("No activation function");
            return;
        }
        formData.append("hidden_layers", hiddenLayers);
        formData.append("iterations", iterations);
        formData.append("activation_function", activationFunction);
        formData.append("training_data_count", trainingDataCount);
        formData.append("image_number", imageNumber);

        // Add file if exists
        if (file) {
            formData.append("file", file);
        }

        try {
            const response = await fetch("http://localhost:5000/process", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                // Log error response
                const errorText = await response.text();
                console.error('Error response:', errorText);
                return;
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("video/mp4")) {
                const videoBlob = await response.blob();
                const videoObjectURL = URL.createObjectURL(videoBlob);
                setVideoUrl(videoObjectURL);
            } else {
                const result = await response.json();
                console.log(result);
            }
        } catch (error) {
            console.error("Error processing request:", error);
        }
    };

    return (
        <div>
            <div className="Header">
                <div className="Title">
                    <img src={logo} alt="logo" width="15%"></img>
                    <div className="NNV">NEURAL NETWORK VISUALIZER</div>
                </div>
                <div className="Contact">
                    <a href="https://github.com/" target="_blank" rel="noopener noreferrer">
                        <FontAwesomeIcon icon={faGithub} size="2x" style={{ color: "#00b3b3" }} />
                    </a>
                    <a href="https://linkedin.com/" target="_blank" rel="noopener noreferrer">
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
                    {videoUrl && (
                        <video
                            width="auto"
                            height="95%"
                            controls
                            src={videoUrl}
                        >
                            Your browser does not support the video tag.
                        </video>
                    )}
                </main>
            </div>
        </div>
    );
}

export default App;