from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for, session
import pymysql
import os
import numpy as np
import librosa
from scipy.spatial.distance import cosine
import speech_recognition as sr
import tempfile
import uuid
import logging
import traceback
import subprocess
from pydub import AudioSegment
import io
import ffmpeg
import json
from textblob import TextBlob
from datetime import datetime, timedelta
import base64
import pickle
from werkzeug.security import generate_password_hash, check_password_hash
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Set up logging with more detailed information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database Connection
def get_db_connection():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="Neha@6283",
        database="voice_biometric"
    )
    return conn

# Update the UPLOAD_DIR to use audio_files
UPLOAD_DIR = "audio_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global quotes list
quotes = []

def load_quotes():
    """Load quotes from JSON file."""
    global quotes
    try:
        # Get the absolute path to quotes.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        quotes_path = os.path.join(current_dir, 'quotes.json')
        logger.info(f"Loading quotes from: {quotes_path}")
        
        if not os.path.exists(quotes_path):
            logger.error(f"quotes.json not found at: {quotes_path}")
            raise FileNotFoundError(f"quotes.json not found at: {quotes_path}")
        
        with open(quotes_path, 'r', encoding='utf-8') as f:
            quotes_data = json.load(f)
            if isinstance(quotes_data, dict) and 'quotes' in quotes_data:
                quotes = quotes_data['quotes']
                logger.info(f"Successfully loaded {len(quotes)} quotes from quotes.json")
                # Log first quote as verification
                if quotes:
                    logger.info(f"First quote loaded: {quotes[0]['quote']}")
            else:
                logger.error("Invalid quotes.json format - missing 'quotes' key")
                quotes = [
                    {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]},
                    {"quote": "Every day is a new beginning.", "author": "Unknown", "tags": ["motivation"]},
                    {"quote": "Life is what happens when you're busy making other plans.", "author": "John Lennon", "tags": ["life"]}
                ]
    except FileNotFoundError as e:
        logger.error(f"quotes.json file not found: {str(e)}")
        quotes = [
            {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]},
            {"quote": "Every day is a new beginning.", "author": "Unknown", "tags": ["motivation"]},
            {"quote": "Life is what happens when you're busy making other plans.", "author": "John Lennon", "tags": ["life"]}
        ]
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing quotes.json: {str(e)}")
        quotes = [
            {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]},
            {"quote": "Every day is a new beginning.", "author": "Unknown", "tags": ["motivation"]},
            {"quote": "Life is what happens when you're busy making other plans.", "author": "John Lennon", "tags": ["life"]}
        ]
    except Exception as e:
        logger.error(f"Error loading quotes: {str(e)}")
        quotes = [
            {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]},
            {"quote": "Every day is a new beginning.", "author": "Unknown", "tags": ["motivation"]},
            {"quote": "Life is what happens when you're busy making other plans.", "author": "John Lennon", "tags": ["life"]}
        ]

# Load quotes at startup
load_quotes()

# Initialize database
def init_db():
    """Initialize the database with required tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(100) NOT NULL,
                user_passphrase VARCHAR(255) NOT NULL,
                voice_features BLOB
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        conn.close()

# Initialize database at startup
init_db()

# Function to convert WebM to WAV if needed
def convert_webm_to_wav(webm_path, wav_path=None):
    """Convert WebM audio to WAV format using ffmpeg."""
    try:
        if wav_path is None:
            wav_path = tempfile.mktemp(suffix='.wav')
        
        # Convert WebM to WAV using ffmpeg with explicit format specification
        stream = ffmpeg.input(webm_path)
        stream = ffmpeg.output(stream, wav_path, 
                             acodec='pcm_s16le',  # Use 16-bit PCM
                             ac=1,                # Mono channel
                             ar='44100',          # Sample rate
                             format='wav')        # Explicitly specify WAV format
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # Verify the output file exists and is not empty
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            logger.error("Failed to create valid WAV file")
            return None
            
        return wav_path
    except Exception as e:
        logger.error(f"Error converting WebM to WAV: {str(e)}")
        return None

# Function to extract features
def extract_features(file_path):
    """Extract voice features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extract pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/10])
        
        # Combine features
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            [pitch_mean]
        ])
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

# Function to compare voice features using Cosine Similarity
def compare_voice(features1, features2, threshold=0.7):
    """Compare two sets of voice features using cosine similarity."""
    try:
        # Ensure features are numpy arrays
        features1 = np.array(features1)
        features2 = np.array(features2)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(features1, features2)
        logger.info(f"Voice similarity score: {similarity}")
        return similarity > threshold, similarity
    except Exception as e:
        logger.error(f"Error comparing voice features: {str(e)}")
        return False, 0

# Function to convert speech to text
def speech_to_text(audio_path):
    """Convert speech to text using Google Speech Recognition."""
    try:
        # First convert to WAV if needed
        if not audio_path.endswith('.wav'):
            wav_path = convert_webm_to_wav(audio_path)
            if not wav_path:
                logger.error("Failed to convert audio to WAV")
                return None
            audio_path = wav_path
        
        recognizer = sr.Recognizer()
        
        # Adjust for ambient noise
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
            
            # Use Google's speech recognition with language model
            text = recognizer.recognize_google(audio, language='en-US')
            logger.info(f"Raw speech recognition: {text}")
            
            # Clean and normalize the text
            text = text.lower().strip()
            # Remove common filler words and normalize
            text = ' '.join([word for word in text.split() if word not in ['is', 'the', 'a', 'an', 'and']])
            logger.info(f"Normalized speech: {text}")
            
            return text
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}")
        return None

# Function to compare passphrases
def compare_passphrases(spoken_text, stored_passphrase):
    """Compare spoken text with stored passphrase with some flexibility."""
    try:
        # Normalize both texts
        spoken = spoken_text.lower().strip()
        stored = stored_passphrase.lower().strip()
        
        # Remove common filler words
        spoken = ' '.join([word for word in spoken.split() if word not in ['is', 'the', 'a', 'an', 'and']])
        stored = ' '.join([word for word in stored.split() if word not in ['is', 'the', 'a', 'an', 'and']])
        
        # Check for exact match
        if spoken == stored:
            return True
            
        # Check for partial match (if one is a subset of the other)
        if spoken in stored or stored in spoken:
            return True
            
        # Calculate similarity ratio
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, spoken, stored).ratio()
        logger.info(f"Passphrase similarity: {similarity}")
        
        # Consider it a match if similarity is above 0.8
        return similarity > 0.8
    except Exception as e:
        logger.error(f"Error comparing passphrases: {str(e)}")
        return False

# Function to fetch user details from DB
def get_user(username):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password, user_passphrase, voice_features FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error retrieving user: {e}")
        return None

# Function to save audio file
def save_audio_file(audio_file, username, mode):
    try:
        if not audio_file:
            logger.error("No audio file provided")
            return None
            
        # Create a unique filename
        unique_filename = f"{username}_{mode}_{uuid.uuid4().hex}.wav"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the audio file
        audio_file.save(file_path)
        logger.info(f"Audio file saved: {file_path}")
        
        return file_path
    except Exception as e:
        logger.error(f"Error saving audio file: {e}")
        return None

# Function to get sentiment
def get_sentiment(text):
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.2:
            return "positive"
        elif polarity < -0.2:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return "neutral"

# Function to get relevant quote
def get_relevant_quote(sentiment):
    """Get a relevant quote based on sentiment."""
    try:
        sentiment_tags = {
            "positive": ["inspiration", "motivation", "hope", "success", "happiness", "positivity"],
            "negative": ["wisdom", "life", "philosophy", "perseverance", "courage"],
            "neutral": ["humor", "science", "self", "thought", "wisdom"]
        }
        
        relevant_tags = sentiment_tags.get(sentiment, [])
        matching_quotes = []
        
        # Ensure quotes is a list and not empty
        if not isinstance(quotes, list) or not quotes:
            logger.error("Quotes is not a valid list")
            return {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]}
            
        for quote in quotes:
            if isinstance(quote, dict) and 'tags' in quote and any(tag in quote['tags'] for tag in relevant_tags):
                matching_quotes.append(quote)
        
        if matching_quotes:
            selected_quote = random.choice(matching_quotes)
            logger.info(f"Selected quote: {selected_quote['quote']}")
            return selected_quote
            
        # If no matching quotes, return a random quote
        selected_quote = random.choice(quotes)
        logger.info(f"Selected random quote: {selected_quote['quote']}")
        return selected_quote
    except Exception as e:
        logger.error(f"Error getting relevant quote: {str(e)}")
        return {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]}

# Routes
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('quotes'))
    return render_template('index.html')

@app.route('/quotes')
def quotes():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('quotes.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        # Get form data
        username = request.form.get('username')
        password = request.form.get('password')
        user_passphrase = request.form.get('user_passphrase')
        audio_file = request.files.get('audio')
        
        if not all([username, password, user_passphrase, audio_file]):
            return "❌ All fields are required!", 400
        
        # Save the uploaded WebM file
        webm_path = tempfile.mktemp(suffix='.webm')
        audio_file.save(webm_path)
        
        try:
            # Convert WebM to WAV
            wav_path = convert_webm_to_wav(webm_path)
            if not wav_path:
                return "❌ Error converting audio file!", 500
            
            # Extract features
            features = extract_features(wav_path)
            if features is None:
                return "❌ Error processing audio features!", 500
            
            # Convert speech to text
            spoken_text = speech_to_text(wav_path)
            if not spoken_text:
                return "❌ Could not recognize speech. Please try again.", 400
            
            # Verify passphrase with flexible matching
            if not compare_passphrases(spoken_text, user_passphrase):
                return f"❌ Spoken text does not match passphrase! You said: '{spoken_text}'", 400
            
            # Store user data in database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return "❌ Username already exists!", 400
            
            # Insert new user
            cursor.execute(
                "INSERT INTO users (username, password, user_passphrase, voice_features) VALUES (%s, %s, %s, %s)",
                (username, password, user_passphrase, pickle.dumps(features))
            )
            conn.commit()
            
            return "✅ User registered successfully!", 200
            
        finally:
            # Clean up temporary files
            if os.path.exists(webm_path):
                os.remove(webm_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"❌ Registration failed: {str(e)}", 500

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    audio_file = request.files.get('audio')
    
    if not all([username, password, audio_file]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Save the uploaded audio file
        audio_path = os.path.join(UPLOAD_DIR, f'{username}_login.webm')
        audio_file.save(audio_path)
        
        try:
            # Convert WebM to WAV
            wav_path = os.path.join(UPLOAD_DIR, f'{username}_login.wav')
            wav_path = convert_webm_to_wav(audio_path, wav_path)
            
            if not wav_path:
                raise ValueError("Failed to convert audio file")
            
            # Extract features from the audio
            features = extract_features(wav_path)
            if features is None:
                raise ValueError("Failed to extract voice features")
            
            # Get user from database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, password, voice_features FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Verify password
            if password != user[2]:
                logger.error(f"Password verification failed for user: {username}")
                return jsonify({'error': 'Invalid password'}), 401
            
            # Compare voice features
            stored_features = pickle.loads(user[3])  # voice_features is at index 3
            is_match, similarity = compare_voice(features, stored_features)
            
            if not is_match:
                logger.error(f"Voice verification failed for user: {username} (similarity: {similarity:.2f})")
                return jsonify({'error': f'Voice verification failed (similarity: {similarity:.2f})'}), 401
            
            # Store user in session
            session['user_id'] = user[0]
            session['username'] = username
            
            # Return success response with redirect URL
            return jsonify({
                'success': True,
                'message': 'Login successful!',
                'redirect': '/quotes'  # Use direct URL instead of url_for
            })
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred during login'}), 500

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        data = request.json
        user_text = data.get('text', '')
        audio_data = data.get('audio', None)
        
        if audio_data:
            # Save audio file temporarily
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(UPLOAD_DIR, f"recording_{timestamp}.webm")
            
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            
            try:
                # Convert WebM to WAV
                wav_path = convert_webm_to_wav(audio_path)
                if not wav_path:
                    return jsonify({'error': 'Could not convert audio to WAV format'}), 400
                
                # Convert speech to text
                user_text = speech_to_text(wav_path)
                if not user_text:
                    return jsonify({'error': 'Could not process audio'}), 400
            finally:
                # Clean up temporary files
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
        
        if not user_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get sentiment and relevant quote
        sentiment = get_sentiment(user_text)
        logger.info(f"Detected sentiment: {sentiment}")
        
        # Ensure quotes are loaded
        if not isinstance(quotes, list) or not quotes:
            logger.error("Quotes not properly loaded, reloading...")
            load_quotes()
            if not isinstance(quotes, list) or not quotes:
                return jsonify({'error': 'Quotes not available'}), 500
            
        quote = get_relevant_quote(sentiment)
        logger.info(f"Selected quote: {quote['quote']}")
        
        return jsonify({
            'sentiment': sentiment,
            'quote': quote['quote'],
            'author': quote['author'],
            'transcribed_text': user_text if audio_data else None
        })
    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

@app.route('/get_passphrase')
def get_passphrase():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_passphrase FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return jsonify({"passphrase": result[0]})
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logger.error(f"Error getting passphrase: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Debug route - validate if routes are registered correctly
@app.route('/debug/routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": str(rule)
        })
    return jsonify({"routes": routes})

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    logger.info(f"Serving static file: {path}")
    return send_from_directory('static', path)

# Serve files from uploads directory
@app.route('/uploads/<path:path>')
def serve_uploads(path):
    logger.info(f"Serving upload file: {path}")
    return send_from_directory(UPLOAD_DIR, path)

# Health check endpoint
@app.route('/health')
def health_check():
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "healthy"}), 200

# Catch-all route for debugging
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def catch_all(path):
    logger.warning(f"Unhandled route accessed: {path}, method: {request.method}")
    logger.warning(f"Headers: {request.headers}")
    return f"Route not found: {path}", 404

if __name__ == '__main__':
    logger.info("Starting Flask application")
    logger.info(f"Available routes: {[str(rule) for rule in app.url_map.iter_rules()]}")
    app.run(debug=True, host='0.0.0.0', port=5000)