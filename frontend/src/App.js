import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Music, Heart, Frown, Meh, Smile, AlertCircle, Play, Loader, ExternalLink, RefreshCw } from 'lucide-react';

const EmotionMusicSystem = () => {
  const [currentView, setCurrentView] = useState('home');
  const [image, setImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [playlists, setPlaylists] = useState([]);
  const [error, setError] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  // Emotion to icon mapping
  const emotionIcons = {
    happy: Smile,
    sad: Frown,
    angry: AlertCircle,
    neutral: Meh,
    surprise: AlertCircle,
    fear: AlertCircle,
    disgust: Frown
  };

  // Emotion to color mapping
  const emotionColors = {
    happy: 'from-yellow-400 to-orange-500',
    sad: 'from-blue-400 to-blue-600',
    angry: 'from-red-500 to-red-700',
    neutral: 'from-gray-400 to-gray-600',
    surprise: 'from-purple-400 to-pink-500',
    fear: 'from-indigo-500 to-purple-700',
    disgust: 'from-green-500 to-teal-600'
  };

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/health`);
      const data = await response.json();
      
      if (data.status === 'healthy') {
        setBackendStatus('connected');
        console.log('✓ Backend connected:', data);
      } else {
        setBackendStatus('error');
        setError('Backend is not responding correctly');
      }
    } catch (err) {
      setBackendStatus('disconnected');
      setError('Cannot connect to backend server. Please ensure Flask server is running on port 5000.');
    }
  };

  // Detect emotion using Flask backend API
  const detectEmotion = async (imageData) => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/detect-emotion`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setDetectedEmotion(data.emotion);
        setConfidence(data.confidence);
        setPlaylists(data.playlists || []);
        setCurrentView('results');
        console.log('✓ Emotion detected:', data.emotion, `(${data.confidence}%)`);
      } else {
        setError(data.error || 'Failed to detect emotion. Please try again.');
        setCurrentView('home');
      }
    } catch (err) {
      console.error('API Error:', err);
      setError(`Backend connection failed: ${err.message}. Please ensure Flask server is running.`);
      setCurrentView('home');
    } finally {
      setIsProcessing(false);
    }
  };

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setShowCamera(true);
        setCurrentView('camera');
        setError(null);
      }
    } catch (err) {
      setError('Could not access camera. Please check permissions.');
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setShowCamera(false);
  };

  // Capture photo from camera
  const capturePhoto = () => {
    if (canvasRef.current && videoRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      const imageData = canvas.toDataURL('image/jpeg');
      setImage(imageData);
      stopCamera();
      detectEmotion(imageData);
    }
  };

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageData = event.target.result;
        setImage(imageData);
        detectEmotion(imageData);
      };
      reader.readAsDataURL(file);
    }
  };

  // Reset to home
  const resetToHome = () => {
    stopCamera();
    setCurrentView('home');
    setImage(null);
    setDetectedEmotion(null);
    setConfidence(0);
    setPlaylists([]);
    setError(null);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Backend Status Indicator
  const BackendStatusBadge = () => (
    <div className="fixed top-4 right-4 z-50">
      <div className={`flex items-center gap-2 px-4 py-2 rounded-full backdrop-blur-lg border ${
        backendStatus === 'connected' 
          ? 'bg-green-500 bg-opacity-20 border-green-400' 
          : backendStatus === 'disconnected'
          ? 'bg-red-500 bg-opacity-20 border-red-400'
          : 'bg-yellow-500 bg-opacity-20 border-yellow-400'
      }`}>
        <div className={`w-2 h-2 rounded-full ${
          backendStatus === 'connected' 
            ? 'bg-green-400 animate-pulse' 
            : backendStatus === 'disconnected'
            ? 'bg-red-400'
            : 'bg-yellow-400 animate-pulse'
        }`} />
        <span className="text-white text-sm font-medium">
          {backendStatus === 'connected' 
            ? 'Backend Connected' 
            : backendStatus === 'disconnected'
            ? 'Backend Offline'
            : 'Checking...'}
        </span>
        {backendStatus !== 'connected' && (
          <button 
            onClick={checkBackendHealth}
            className="ml-2 hover:bg-white hover:bg-opacity-20 p-1 rounded"
          >
            <RefreshCw className="w-3 h-3 text-white" />
          </button>
        )}
      </div>
    </div>
  );

  // Home View
  if (currentView === 'home') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 flex items-center justify-center p-6">
        <BackendStatusBadge />
        
        <div className="max-w-2xl w-full">
          <div className="text-center mb-12">
            <Music className="w-20 h-20 mx-auto mb-6 text-white animate-bounce" />
            <h1 className="text-5xl font-bold text-white mb-4">
              Emotion Music Recommender
            </h1>
            <p className="text-xl text-indigo-200">
              AI-powered emotion detection with personalized music recommendations
            </p>
          </div>

          {error && (
            <div className="bg-red-500 bg-opacity-20 border border-red-400 text-white px-4 py-3 rounded-lg mb-6 flex items-start">
              <AlertCircle className="w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="font-semibold mb-1">Connection Error</p>
                <p className="text-sm">{error}</p>
                {backendStatus === 'disconnected' && (
                  <p className="text-sm mt-2 text-red-200">
                    💡 Run: <code className="bg-black bg-opacity-30 px-2 py-1 rounded">python app.py</code>
                  </p>
                )}
              </div>
            </div>
          )}

          <div className="grid md:grid-cols-2 gap-6">
            <button
              onClick={startCamera}
              disabled={backendStatus !== 'connected'}
              className={`rounded-2xl p-8 transition-all duration-300 transform ${
                backendStatus === 'connected'
                  ? 'bg-white bg-opacity-10 hover:bg-opacity-20 hover:scale-105 hover:shadow-2xl cursor-pointer'
                  : 'bg-white bg-opacity-5 cursor-not-allowed opacity-50'
              } backdrop-blur-lg border border-white border-opacity-20`}
            >
              <Camera className="w-16 h-16 mx-auto mb-4 text-white" />
              <h3 className="text-2xl font-semibold text-white mb-2">
                Use Camera
              </h3>
              <p className="text-indigo-200">
                Capture your facial expression in real-time
              </p>
            </button>

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={backendStatus !== 'connected'}
              className={`rounded-2xl p-8 transition-all duration-300 transform ${
                backendStatus === 'connected'
                  ? 'bg-white bg-opacity-10 hover:bg-opacity-20 hover:scale-105 hover:shadow-2xl cursor-pointer'
                  : 'bg-white bg-opacity-5 cursor-not-allowed opacity-50'
              } backdrop-blur-lg border border-white border-opacity-20`}
            >
              <Upload className="w-16 h-16 mx-auto mb-4 text-white" />
              <h3 className="text-2xl font-semibold text-white mb-2">
                Upload Photo
              </h3>
              <p className="text-indigo-200">
                Choose an image from your device
              </p>
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
            disabled={backendStatus !== 'connected'}
          />

          <div className="mt-12 text-center text-indigo-300 text-sm space-y-2">
            <p>Powered by TensorFlow & Spotify API</p>
            <p className="text-xs">
              Supports 7 emotions: Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Camera View
  if (currentView === 'camera') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 flex items-center justify-center p-6">
        <BackendStatusBadge />
        
        <div className="max-w-3xl w-full">
          <div className="bg-white bg-opacity-10 backdrop-blur-lg border border-white border-opacity-20 rounded-2xl p-6">
            <h2 className="text-3xl font-bold text-white mb-6 text-center">
              Position Your Face
            </h2>
            
            <div className="relative rounded-xl overflow-hidden mb-6">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full rounded-xl"
              />
              <div className="absolute inset-0 border-4 border-white border-opacity-50 rounded-xl pointer-events-none" />
              <div className="absolute bottom-4 left-0 right-0 text-center">
                <p className="text-white text-sm bg-black bg-opacity-50 inline-block px-4 py-2 rounded-full">
                  Make sure your face is clearly visible and well-lit
                </p>
              </div>
            </div>

            <canvas ref={canvasRef} className="hidden" />

            <div className="flex gap-4">
              <button
                onClick={capturePhoto}
                className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white py-4 rounded-xl font-semibold transition-all duration-300 transform hover:scale-105 flex items-center justify-center"
              >
                <Camera className="w-5 h-5 mr-2" />
                Capture Photo
              </button>
              
              <button
                onClick={resetToHome}
                className="flex-1 bg-white bg-opacity-10 hover:bg-opacity-20 text-white py-4 rounded-xl font-semibold transition-all duration-300"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Processing View
  if (isProcessing) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 flex items-center justify-center p-6">
        <div className="text-center">
          <Loader className="w-16 h-16 text-white animate-spin mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-white mb-2">
            Analyzing Your Emotion...
          </h2>
          <p className="text-indigo-300 mb-4">
            Using deep learning to detect your emotion
          </p>
          <div className="text-indigo-400 text-sm">
            This may take a few seconds
          </div>
        </div>
      </div>
    );
  }

  // Results View
  if (currentView === 'results') {
    const EmotionIcon = emotionIcons[detectedEmotion] || Meh;
    const emotionColor = emotionColors[detectedEmotion] || 'from-gray-400 to-gray-600';
    
    // Flatten all tracks from all playlists
    const allTracks = playlists.flatMap(playlist => 
      playlist.tracks.map(track => ({
        ...track,
        playlistName: playlist.playlist_name,
        playlistUrl: playlist.playlist_url
      }))
    );

    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 p-6 overflow-y-auto">
        <BackendStatusBadge />
        
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">
              Emotion Detected!
            </h1>
            <button
              onClick={resetToHome}
              className="text-indigo-300 hover:text-white transition-colors"
            >
              ← Try Another Photo
            </button>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Emotion Result Card */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg border border-white border-opacity-20 rounded-2xl p-8">
              <h2 className="text-2xl font-semibold text-white mb-6">
                Analysis Results
              </h2>
              
              {image && (
                <div className="mb-6 rounded-xl overflow-hidden border-2 border-white border-opacity-20">
                  <img 
                    src={image} 
                    alt="Captured face" 
                    className="w-full h-64 object-cover"
                  />
                </div>
              )}

              <div className={`bg-gradient-to-r ${emotionColor} rounded-xl p-6 text-center mb-4`}>
                <EmotionIcon className="w-16 h-16 mx-auto mb-4 text-white" />
                <h3 className="text-3xl font-bold text-white capitalize mb-2">
                  {detectedEmotion}
                </h3>
                <div className="text-white text-opacity-90">
                  Confidence: {confidence.toFixed(1)}%
                </div>
              </div>

              <div className="bg-white bg-opacity-5 rounded-lg p-4">
                <div className="flex justify-between text-sm text-indigo-300 mb-2">
                  <span>Confidence Level</span>
                  <span>{confidence.toFixed(1)}%</span>
                </div>
                <div className="bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div 
                    className={`bg-gradient-to-r ${emotionColor} h-full transition-all duration-1000`}
                    style={{ width: `${confidence}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Playlist Card */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg border border-white border-opacity-20 rounded-2xl p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold text-white">
                  Recommended Music
                </h2>
                <Music className="w-6 h-6 text-indigo-300" />
              </div>

              <div className="space-y-3 max-h-96 overflow-y-auto">
                {allTracks.slice(0, 6).map((track, index) => (
                  <div 
                    key={index}
                    className="bg-white bg-opacity-5 hover:bg-opacity-10 rounded-lg p-4 transition-all duration-300 transform hover:scale-102 cursor-pointer group"
                  >
                    <div className="flex items-center gap-4">
                      <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg p-3 group-hover:scale-110 transition-transform flex-shrink-0">
                        {track.album_art ? (
                          <img src={track.album_art} alt="Album" className="w-10 h-10 rounded" />
                        ) : (
                          <Play className="w-5 h-5 text-white" />
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <h4 className="text-white font-semibold truncate">
                          {track.title}
                        </h4>
                        <p className="text-indigo-300 text-sm truncate">
                          {track.artist}
                        </p>
                      </div>
                      
                      <div className="text-right flex-shrink-0">
                        <div className="text-indigo-300 text-sm">
                          {track.duration}
                        </div>
                        <div className="text-indigo-400 text-xs">
                          {track.genre}
                        </div>
                      </div>

                      {track.spotify_url && track.spotify_url !== '#' && (
                        <a
                          href={track.spotify_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-green-400 hover:text-green-300 transition-colors"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {playlists.length > 0 && playlists[0].playlist_url !== '#' && (
                <a
                  href={playlists[0].playlist_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full mt-6 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white py-4 rounded-xl font-semibold transition-all duration-300 transform hover:scale-105 flex items-center justify-center"
                >
                  <Music className="w-5 h-5 mr-2" />
                  Open Full Playlist in Spotify
                </a>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default EmotionMusicSystem;