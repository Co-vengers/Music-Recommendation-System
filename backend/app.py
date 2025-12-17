from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
from io import BytesIO
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = 'models/emotion_model.h5'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# CRITICAL: Emotion labels MUST match the order from training
# This matches the folder order: angry, disgust, fear, happy, neutral, sad, surprise
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Spotify credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'your_client_id')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', 'your_client_secret')

# Global variables
emotion_model = None
face_cascade = None
spotify_client = None

# Emotion to Spotify mood mapping
EMOTION_TO_MOOD = {
    'happy': ['happy', 'party', 'summer'],
    'sad': ['sad', 'chill', 'rainy-day'],
    'angry': ['metal', 'rock', 'power'],
    'neutral': ['study', 'ambient', 'focus'],
    'surprise': ['dance', 'pop', 'exciting'],
    'fear': ['calm', 'peaceful', 'acoustic'],
    'disgust': ['indie', 'alternative', 'fresh']
}

def load_models():
    """Load the emotion detection model and face cascade"""
    global emotion_model, face_cascade, spotify_client
    
    try:
        # Load emotion detection model
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading emotion model from {MODEL_PATH}...")
            emotion_model = load_model(MODEL_PATH)
            logger.info(f"✓ Emotion model loaded successfully")
            
            # Verify model input shape
            expected_shape = (None, 48, 48, 1)
            actual_shape = emotion_model.input_shape
            logger.info(f"  Model input shape: {actual_shape}")
            logger.info(f"  Model output shape: {emotion_model.output_shape}")
            
            if actual_shape[1:] != expected_shape[1:]:
                logger.warning(f"  ⚠ Model shape mismatch! Expected {expected_shape}, got {actual_shape}")
        else:
            logger.warning(f"⚠ Model not found at {MODEL_PATH}")
            logger.warning(f"  Please train the model first using: python train_model.py")
            logger.warning(f"  Using fallback random detection for testing.")
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
        logger.info("✓ Face cascade classifier loaded")
        
        # Initialize Spotify client
        if SPOTIFY_CLIENT_ID != 'your_client_id' and SPOTIFY_CLIENT_SECRET != 'your_client_secret':
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET
                )
                spotify_client = spotipy.Spotify(auth_manager=auth_manager)
                # Test connection
                spotify_client.search(q='test', type='playlist', limit=1)
                logger.info("✓ Spotify client initialized and connected")
            except Exception as e:
                logger.warning(f"⚠ Spotify connection failed: {str(e)}")
                logger.warning("  Using fallback playlists")
                spotify_client = None
        else:
            logger.warning("⚠ Spotify credentials not configured")
            logger.warning("  Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
            logger.warning("  Using fallback playlists")
        
    except Exception as e:
        logger.error(f"✗ Error loading models: {str(e)}")
        raise

def decode_base64_image(base64_string):
    """Decode base64 image to OpenCV format"""
    try:
        # Remove header if present (data:image/png;base64,...)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Image decoded: shape={image_cv.shape}, dtype={image_cv.dtype}")
        return image_cv
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        raise Exception(f"Failed to decode image: {str(e)}")

def detect_face(image):
    """Detect face in the image using Haar Cascade"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with multiple scales
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            logger.warning("No face detected in image")
            return None
        
        # Return the largest face (most likely to be the primary subject)
        faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        largest_face = faces_sorted[0]
        
        logger.info(f"Detected {len(faces)} face(s), using largest: {largest_face}")
        return largest_face
    
    except Exception as e:
        logger.error(f"Face detection error: {str(e)}")
        return None

def preprocess_face(face_img):
    """Preprocess face for emotion detection - MUST match training preprocessing"""
    try:
        # Resize to 48x48 (standard for emotion detection models)
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert to grayscale (model expects single channel)
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values to [0, 1] - CRITICAL: must match training
        face_img = face_img.astype('float32') / 255.0
        
        # Reshape for model input: (1, 48, 48, 1)
        # batch_size=1, height=48, width=48, channels=1
        face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
        face_img = np.expand_dims(face_img, axis=0)   # Add batch dimension
        
        logger.info(f"Preprocessed face shape: {face_img.shape}, range: [{face_img.min():.3f}, {face_img.max():.3f}]")
        return face_img
    
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

def predict_emotion(face_img):
    """Predict emotion using the trained CNN model"""
    if emotion_model is None:
        # Fallback: random prediction for testing without model
        logger.warning("Using fallback random prediction (model not loaded)")
        import random
        emotion_idx = random.randint(0, len(EMOTION_LABELS) - 1)
        confidence = 65 + random.random() * 30
        return EMOTION_LABELS[emotion_idx], confidence
    
    try:
        # Predict emotion
        predictions = emotion_model.predict(face_img, verbose=0)
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx]) * 100
        emotion = EMOTION_LABELS[emotion_idx]
        
        # Log all predictions for debugging
        prediction_str = ", ".join([f"{EMOTION_LABELS[i]}: {predictions[0][i]*100:.1f}%" 
                                   for i in range(len(EMOTION_LABELS))])
        logger.info(f"Predictions: {prediction_str}")
        logger.info(f"Detected emotion: {emotion} ({confidence:.2f}%)")
        
        return emotion, confidence
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return neutral as fallback
        return 'neutral', 50.0

def get_spotify_playlists(emotion):
    """Get Spotify playlists based on detected emotion"""
    if spotify_client is None:
        logger.info(f"Using fallback playlists for emotion: {emotion}")
        return get_fallback_playlists(emotion)
    
    try:
        moods = EMOTION_TO_MOOD.get(emotion, ['chill'])
        playlists = []
        
        # Search for playlists matching the mood
        for mood in moods[:2]:  # Use first 2 moods
            results = spotify_client.search(q=mood, type='playlist', limit=3)
            
            for playlist in results['playlists']['items']:
                try:
                    # Get playlist tracks
                    tracks_result = spotify_client.playlist_tracks(
                        playlist['id'], 
                        limit=5
                    )
                    
                    tracks = []
                    for item in tracks_result['items']:
                        if item['track']:
                            track = item['track']
                            tracks.append({
                                'title': track['name'],
                                'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                                'duration': format_duration(track['duration_ms']),
                                'genre': mood.capitalize(),
                                'spotify_url': track['external_urls']['spotify'],
                                'preview_url': track.get('preview_url'),
                                'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None
                            })
                    
                    if tracks:
                        playlists.append({
                            'playlist_name': playlist['name'],
                            'playlist_url': playlist['external_urls']['spotify'],
                            'tracks': tracks
                        })
                        
                        if len(playlists) >= 2:
                            break
                except Exception as track_error:
                    logger.warning(f"Error fetching tracks: {str(track_error)}")
                    continue
            
            if len(playlists) >= 2:
                break
        
        if playlists:
            logger.info(f"Found {len(playlists)} Spotify playlists for {emotion}")
            return playlists
        else:
            logger.info(f"No Spotify playlists found, using fallback for {emotion}")
            return get_fallback_playlists(emotion)
    
    except Exception as e:
        logger.error(f"Spotify API error: {str(e)}")
        return get_fallback_playlists(emotion)

def get_fallback_playlists(emotion):
    """Return curated fallback playlists when Spotify API is unavailable"""
    fallback_data = {
        'happy': {
            'playlist_name': '😊 Happy Vibes Collection',
            'tracks': [
                {'title': 'Happy', 'artist': 'Pharrell Williams', 'duration': '3:53', 'genre': 'Pop'},
                {'title': 'Good Vibrations', 'artist': 'The Beach Boys', 'duration': '3:36', 'genre': 'Rock'},
                {'title': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'duration': '3:59', 'genre': 'Pop'},
                {'title': 'Don\'t Stop Me Now', 'artist': 'Queen', 'duration': '3:29', 'genre': 'Rock'},
                {'title': 'I Gotta Feeling', 'artist': 'The Black Eyed Peas', 'duration': '4:49', 'genre': 'Pop'}
            ]
        },
        'sad': {
            'playlist_name': '😢 Melancholic Moments',
            'tracks': [
                {'title': 'Someone Like You', 'artist': 'Adele', 'duration': '4:45', 'genre': 'Pop'},
                {'title': 'The Night We Met', 'artist': 'Lord Huron', 'duration': '3:28', 'genre': 'Indie'},
                {'title': 'Skinny Love', 'artist': 'Bon Iver', 'duration': '3:58', 'genre': 'Indie'},
                {'title': 'Fix You', 'artist': 'Coldplay', 'duration': '4:54', 'genre': 'Alternative'},
                {'title': 'Hurt', 'artist': 'Johnny Cash', 'duration': '3:38', 'genre': 'Country'}
            ]
        },
        'angry': {
            'playlist_name': '😠 Rage Release',
            'tracks': [
                {'title': 'In The End', 'artist': 'Linkin Park', 'duration': '3:36', 'genre': 'Rock'},
                {'title': 'Break Stuff', 'artist': 'Limp Bizkit', 'duration': '2:46', 'genre': 'Metal'},
                {'title': 'Killing In The Name', 'artist': 'Rage Against The Machine', 'duration': '5:13', 'genre': 'Metal'},
                {'title': 'Bodies', 'artist': 'Drowning Pool', 'duration': '3:22', 'genre': 'Metal'},
                {'title': 'Down With The Sickness', 'artist': 'Disturbed', 'duration': '4:38', 'genre': 'Metal'}
            ]
        },
        'neutral': {
            'playlist_name': '😐 Focus & Flow',
            'tracks': [
                {'title': 'Weightless', 'artist': 'Marconi Union', 'duration': '8:09', 'genre': 'Ambient'},
                {'title': 'Clair de Lune', 'artist': 'Claude Debussy', 'duration': '5:03', 'genre': 'Classical'},
                {'title': 'Minecraft', 'artist': 'C418', 'duration': '4:14', 'genre': 'Ambient'},
                {'title': 'Spiegel im Spiegel', 'artist': 'Arvo Pärt', 'duration': '8:36', 'genre': 'Classical'},
                {'title': 'Porcelain', 'artist': 'Moby', 'duration': '4:01', 'genre': 'Electronic'}
            ]
        },
        'surprise': {
            'playlist_name': '😲 Unexpected Beats',
            'tracks': [
                {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'duration': '4:30', 'genre': 'Funk'},
                {'title': 'Shut Up and Dance', 'artist': 'WALK THE MOON', 'duration': '3:19', 'genre': 'Pop'},
                {'title': 'Can\'t Stop The Feeling!', 'artist': 'Justin Timberlake', 'duration': '3:56', 'genre': 'Pop'},
                {'title': 'Electric Feel', 'artist': 'MGMT', 'duration': '3:49', 'genre': 'Indie'},
                {'title': 'Wake Me Up', 'artist': 'Avicii', 'duration': '4:09', 'genre': 'EDM'}
            ]
        },
        'fear': {
            'playlist_name': '😨 Courage & Calm',
            'tracks': [
                {'title': 'Brave', 'artist': 'Sara Bareilles', 'duration': '3:40', 'genre': 'Pop'},
                {'title': 'Stronger', 'artist': 'Kelly Clarkson', 'duration': '3:42', 'genre': 'Pop'},
                {'title': 'Eye of the Tiger', 'artist': 'Survivor', 'duration': '4:05', 'genre': 'Rock'},
                {'title': 'Fight Song', 'artist': 'Rachel Platten', 'duration': '3:23', 'genre': 'Pop'},
                {'title': 'Hall of Fame', 'artist': 'The Script ft. will.i.am', 'duration': '3:22', 'genre': 'Pop'}
            ]
        },
        'disgust': {
            'playlist_name': '🤢 Fresh Start',
            'tracks': [
                {'title': 'Clean', 'artist': 'Taylor Swift', 'duration': '4:31', 'genre': 'Pop'},
                {'title': 'New Rules', 'artist': 'Dua Lipa', 'duration': '3:29', 'genre': 'Pop'},
                {'title': 'Shake It Off', 'artist': 'Taylor Swift', 'duration': '3:39', 'genre': 'Pop'},
                {'title': 'Problem', 'artist': 'Ariana Grande', 'duration': '3:14', 'genre': 'Pop'},
                {'title': 'Sorry Not Sorry', 'artist': 'Demi Lovato', 'duration': '3:23', 'genre': 'Pop'}
            ]
        }
    }
    
    data = fallback_data.get(emotion, fallback_data['neutral'])
    return [{
        'playlist_name': data['playlist_name'],
        'playlist_url': '#',
        'tracks': data['tracks']
    }]

def format_duration(ms):
    """Convert milliseconds to MM:SS format"""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': emotion_model is not None,
        'face_cascade_loaded': face_cascade is not None,
        'spotify_connected': spotify_client is not None,
        'supported_emotions': EMOTION_LABELS,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion_endpoint():
    """Main endpoint for emotion detection and music recommendation"""
    try:
        # Get image from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided in request body'
            }), 400
        
        logger.info("Received emotion detection request")
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Detect face
        face_coords = detect_face(image)
        if face_coords is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image. Please ensure your face is clearly visible.'
            }), 400
        
        # Extract face region
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess face
        face_processed = preprocess_face(face_roi)
        
        # Predict emotion
        emotion, confidence = predict_emotion(face_processed)
        
        # Get music recommendations
        playlists = get_spotify_playlists(emotion)
        
        # Prepare response
        response = {
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'face_location': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'playlists': playlists,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully processed request: {emotion} ({confidence:.2f}%)")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in detect_emotion_endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@app.route('/api/emotions', methods=['GET'])
def get_emotion_list():
    """Get list of supported emotions"""
    return jsonify({
        'emotions': EMOTION_LABELS,
        'count': len(EMOTION_LABELS),
        'descriptions': {
            'angry': 'Feeling mad or frustrated',
            'disgust': 'Feeling repulsed or averse',
            'fear': 'Feeling scared or anxious',
            'happy': 'Feeling joyful or content',
            'neutral': 'Feeling calm or indifferent',
            'sad': 'Feeling down or melancholic',
            'surprise': 'Feeling shocked or amazed'
        }
    })

@app.route('/api/test-spotify', methods=['GET'])
def test_spotify():
    """Test Spotify connection"""
    try:
        if spotify_client is None:
            return jsonify({
                'connected': False,
                'message': 'Spotify credentials not configured. Using fallback playlists.',
                'instructions': 'Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables'
            })
        
        # Test search
        results = spotify_client.search(q='happy', type='playlist', limit=1)
        
        return jsonify({
            'connected': True,
            'message': 'Spotify API working correctly',
            'test_result': len(results['playlists']['items']) > 0
        })
    except Exception as e:
        return jsonify({
            'connected': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  FACIAL EMOTION RECOGNITION & MUSIC RECOMMENDATION SYSTEM")
    print("="*70)
    print("\n🚀 Initializing backend server...\n")
    
    # Load models on startup
    try:
        load_models()
        print("\n✓ All components loaded successfully")
    except Exception as e:
        print(f"\n✗ Error during initialization: {str(e)}")
        print("  Some features may not work correctly")
    
    print("\n" + "="*70)
    print("🌐 Server starting on http://localhost:5000")
    print("="*70)
    print("\n📋 Available endpoints:")
    print("  GET  /health              - Health check & system status")
    print("  POST /api/detect-emotion  - Detect emotion from image")
    print("  GET  /api/emotions        - Get supported emotions list")
    print("  GET  /api/test-spotify    - Test Spotify connection")
    print("\n" + "="*70)
    print("\n💡 Tips:")
    print("  • Ensure good lighting for better face detection")
    print("  • Model file: models/emotion_model.h5")
    print("  • Train model first if not available: python train_model.py")
    print("\n" + "="*70 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)