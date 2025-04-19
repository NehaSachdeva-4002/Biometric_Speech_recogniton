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
from modules.quote_recommendation import get_quote_and_suggestion

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
    """Create and return a database connection."""
    try:
        conn = pymysql.connect(
            host="localhost",
            user="root",
            password="Neha@6283",
            database="voice_biometric"
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return None

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
                voice_features BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    """Extract enhanced voice features with gender-specific analysis."""
    try:
        # Load audio with higher sample rate
        y, sr = librosa.load(file_path, sr=44100)
        
        # Anti-spoofing: Check audio duration
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1.0 or duration > 10.0:
            logger.warning(f"Invalid audio duration: {duration} seconds")
            return None
            
        # Anti-spoofing: Check signal quality
        signal_quality = np.mean(np.abs(y))
        if signal_quality < 0.01:
            logger.warning(f"Poor signal quality: {signal_quality}")
            return None
            
        # Enhanced pre-emphasis
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # Extract fundamental frequency (F0) with gender-specific ranges
        pitches, magnitudes = librosa.piptrack(
            y=y, 
            sr=sr,
            fmin=librosa.note_to_hz('C2'),  # Lower range for male voices
            fmax=librosa.note_to_hz('C7'),  # Higher range for female voices
            hop_length=512,
            n_fft=2048
        )
        
        # Filter weak pitch estimates
        pitches = pitches[magnitudes > np.max(magnitudes)/10]
        pitches = pitches[pitches > 0]
        
        if len(pitches) < 10:
            logger.warning("Insufficient pitch points detected")
            return None
            
        # Calculate detailed F0 statistics
        f0_mean = np.mean(pitches)
        f0_std = np.std(pitches)
        f0_range = np.max(pitches) - np.min(pitches)
        f0_median = np.median(pitches)
        f0_skew = np.mean((pitches - f0_mean) ** 3) / (f0_std ** 3)  # Pitch distribution skewness
        
        # Calculate F0 contour with gender-specific analysis
        num_segments = 10
        segment_size = len(pitches) // num_segments
        f0_contour = np.zeros(num_segments)
        f0_variation = np.zeros(num_segments - 1)
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            if end_idx <= len(pitches):
                f0_contour[i] = np.mean(pitches[start_idx:end_idx])
            else:
                f0_contour[i] = np.mean(pitches[start_idx:])
            
            if i > 0:
                f0_variation[i-1] = abs(f0_contour[i] - f0_contour[i-1])
        
        # Extract enhanced formant frequencies with gender-specific analysis
        formants = []
        formant_stability = []
        formant_ratios = []
        
        for i in range(1, 5):
            formant = librosa.effects.preemphasis(y, coef=0.97)
            formant = librosa.effects.preemphasis(formant, coef=0.97)
            formant_mean = np.mean(formant)
            formant_std = np.std(formant)
            formants.append(formant_mean)
            formant_stability.append(formant_std)
            
            if i > 1:
                formant_ratios.append(formants[i-1] / formants[i-2])
        
        # Extract enhanced MFCC features with gender-specific emphasis
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=26,
            n_fft=2048,
            hop_length=512,
            win_length=1024,
            window='hann',
            lifter=0.6
        )
        
        # Extract delta and delta-delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Extract spectral features with gender-specific analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        # Extract harmonic and noise components
        y_harmonic, y_percussive = librosa.effects.hpss(
            y,
            kernel_size=31,
            power=2.0,
            margin=1.0
        )
        
        # Calculate detailed harmonic analysis
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        harmonic_ratio = harmonic_energy / (percussive_energy + 1e-6)
        
        # Extract voice quality features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        rms_energy = librosa.feature.rms(y=y)[0]
        
        # Calculate statistics for each feature
        mfcc_stats = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.max(mfcc, axis=1),
            np.min(mfcc, axis=1)
        ])
        
        mfcc_delta_stats = np.concatenate([
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta, axis=1)
        ])
        
        mfcc_delta2_stats = np.concatenate([
            np.mean(mfcc_delta2, axis=1),
            np.std(mfcc_delta2, axis=1)
        ])
        
        # Combine all features
        features = np.concatenate([
            mfcc_stats,  # 104 features
            mfcc_delta_stats,  # 52 features
            mfcc_delta2_stats,  # 52 features
            [f0_mean, f0_std, f0_range, f0_median, f0_skew],  # Enhanced F0 features
            f0_contour,  # F0 contour
            f0_variation,  # F0 variation
            formants,  # Formant frequencies
            formant_stability,  # Formant stability
            formant_ratios,  # Formant ratios
            [harmonic_ratio],  # Harmonic ratio
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_contrast), np.std(spectral_contrast)],
            [np.mean(spectral_flatness), np.std(spectral_flatness)],
            [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)],
            [np.mean(rms_energy), np.std(rms_energy)]
        ])
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-6)
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

# Function to compare voice features using Cosine Similarity
def compare_voice(features1, features2):
    """Compare voice features with enhanced gender-specific analysis."""
    try:
        if features1 is None or features2 is None:
            logger.error("One or both feature sets are None")
            return 0.0
            
        features1 = np.array(features1).flatten()
        features2 = np.array(features2).flatten()
        
        if features1.shape != features2.shape:
            logger.error(f"Feature shape mismatch: {features1.shape} != {features2.shape}")
            return 0.0
            
        # Normalize features
        features1 = (features1 - np.mean(features1)) / (np.std(features1) + 1e-6)
        features2 = (features2 - np.mean(features2)) / (np.std(features2) + 1e-6)
        
        # Enhanced weights for gender-specific features
        weights = {
            'mfcc': 0.20,          # MFCC features
            'mfcc_delta': 0.15,    # Delta MFCC features
            'mfcc_delta2': 0.10,   # Delta-delta MFCC features
            'f0': 0.25,            # Fundamental frequency (increased weight)
            'formants': 0.15,      # Formant frequencies (increased weight)
            'spectral': 0.10,      # Spectral features
            'voice_quality': 0.05  # Voice quality features
        }
        
        # Calculate similarities for each feature group
        mfcc_sim = 1 - cosine(features1[:104], features2[:104])
        mfcc_delta_sim = 1 - cosine(features1[104:156], features2[104:156])
        mfcc_delta2_sim = 1 - cosine(features1[156:208], features2[156:208])
        f0_sim = 1 - cosine(features1[208:213], features2[208:213])  # Enhanced F0 features
        f0_contour_sim = 1 - cosine(features1[213:223], features2[213:223])
        f0_variation_sim = 1 - cosine(features1[223:232], features2[223:232])
        formant_sim = 1 - cosine(features1[232:236], features2[232:236])
        formant_stability_sim = 1 - cosine(features1[236:240], features2[236:240])
        formant_ratio_sim = 1 - cosine(features1[240:243], features2[240:243])
        spectral_sim = 1 - cosine(features1[243:251], features2[243:251])
        voice_quality_sim = 1 - cosine(features1[251:], features2[251:])
        
        # Calculate weighted final similarity with gender-specific emphasis
        final_similarity = (
            weights['mfcc'] * mfcc_sim +
            weights['mfcc_delta'] * mfcc_delta_sim +
            weights['mfcc_delta2'] * mfcc_delta2_sim +
            weights['f0'] * (f0_sim * 0.5 + f0_contour_sim * 0.3 + f0_variation_sim * 0.2) +
            weights['formants'] * (formant_sim * 0.4 + formant_stability_sim * 0.3 + formant_ratio_sim * 0.3) +
            weights['spectral'] * spectral_sim +
            weights['voice_quality'] * voice_quality_sim
        )
        
        # Anti-spoofing: Check individual feature similarities
        if mfcc_sim < 0.85 or f0_sim < 0.85 or formant_sim < 0.85:
            logger.warning(f"Low similarity in critical features - "
                         f"MFCC: {mfcc_sim:.6f}, "
                         f"F0: {f0_sim:.6f}, "
                         f"Formants: {formant_sim:.6f}")
            return 0.0
            
        logger.info(f"Similarity scores - "
                   f"MFCC: {mfcc_sim:.6f}, "
                   f"MFCC Delta: {mfcc_delta_sim:.6f}, "
                   f"MFCC Delta2: {mfcc_delta2_sim:.6f}, "
                   f"F0: {f0_sim:.6f}, "
                   f"F0 Contour: {f0_contour_sim:.6f}, "
                   f"F0 Variation: {f0_variation_sim:.6f}, "
                   f"Formants: {formant_sim:.6f}, "
                   f"Formant Stability: {formant_stability_sim:.6f}, "
                   f"Formant Ratios: {formant_ratio_sim:.6f}, "
                   f"Spectral: {spectral_sim:.6f}, "
                   f"Voice Quality: {voice_quality_sim:.6f}, "
                   f"Final: {final_similarity:.6f}")
        
        return float(final_similarity)
    except Exception as e:
        logger.error(f"Error comparing voice features: {str(e)}")
        return 0.0

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
    """Register a new user with voice biometrics."""
    try:
        # Get form data
        username = request.form.get('username')
        password = request.form.get('password')
        user_passphrase = request.form.get('user_passphrase')
        audio_file = request.files.get('audio')
        
        logger.info(f"Registration attempt for user: {username}")
        
        if not all([username, password, user_passphrase, audio_file]):
            logger.error("Missing required fields in registration")
            return "❌ All fields are required!", 400
        
        # Check if username already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            logger.error(f"Username already exists: {username}")
            return "❌ Username already exists!", 400
        
        # Save the uploaded WebM file
        webm_path = tempfile.mktemp(suffix='.webm')
        audio_file.save(webm_path)
        
        try:
            # Convert WebM to WAV
            wav_path = convert_webm_to_wav(webm_path)
            if not wav_path:
                logger.error("Failed to convert audio to WAV format")
                return "❌ Error converting audio file!", 500
            
            # Extract features
            features = extract_features(wav_path)
            if features is None:
                logger.error("Failed to extract voice features")
                return "❌ Error processing audio features!", 500
            
            # Convert speech to text
            spoken_text = speech_to_text(wav_path)
            if not spoken_text:
                logger.error("Failed to recognize speech")
                return "❌ Could not recognize speech. Please try again.", 400
            
            # Verify passphrase with flexible matching
            if not compare_passphrases(spoken_text, user_passphrase):
                logger.error(f"Passphrase mismatch - Spoken: '{spoken_text}', Expected: '{user_passphrase}'")
                return f"❌ Spoken text does not match passphrase! You said: '{spoken_text}'", 400
            
            # Store user data in database
            cursor.execute(
                "INSERT INTO users (username, password, user_passphrase, voice_features) VALUES (%s, %s, %s, %s)",
                (username, password, user_passphrase, pickle.dumps(features))
            )
            conn.commit()
            
            logger.info(f"Successfully registered user: {username}")
            return "✅ User registered successfully!", 200
            
        except Exception as e:
            logger.error(f"Error during registration process: {str(e)}")
            return f"❌ Registration failed: {str(e)}", 500
            
        finally:
            # Clean up temporary files
            if os.path.exists(webm_path):
                os.remove(webm_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            conn.close()
            
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return f"❌ Registration failed: {str(e)}", 500

@app.route('/login', methods=['POST'])
def login():
    """Handle user login with voice authentication."""
    try:
        if request.method != 'POST':
            return jsonify({'success': False, 'error': 'Method not allowed'}), 405
            
        # Get form data
        username = request.form.get('username')
        password = request.form.get('password')
        audio_file = request.files.get('audio')
        
        if not all([username, password, audio_file]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
            
        # Verify password first
        if not verify_password(username, password):
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
            
        # Get user's passphrase
        user_passphrase = get_user_passphrase(username)
        if not user_passphrase:
            return jsonify({'success': False, 'error': 'User not found'}), 404
            
        # Save audio file temporarily
        temp_audio_path = os.path.join(tempfile.gettempdir(), f'login_audio_{uuid.uuid4().hex}.webm')
        audio_file.save(temp_audio_path)
        
        try:
            # Extract features from login audio
            login_features = extract_features(temp_audio_path)
            if login_features is None:
                return jsonify({'success': False, 'error': 'Voice authentication failed: Invalid audio quality'}), 400
            
            # Get stored features for the user
            stored_features = get_user_voice_features(username)
            if stored_features is None:
                return jsonify({'success': False, 'error': 'Voice features not found'}), 404
            
            # Compare voice features
            similarity = compare_voice(login_features, stored_features)
            logger.info(f"Voice similarity: {similarity:.6f}")
            
            # Perform speech recognition on the login audio
            login_text = recognize_speech(temp_audio_path)
            if login_text is None:
                return jsonify({'success': False, 'error': 'Could not recognize speech'}), 400
                
            logger.info(f"Recognized login speech: {login_text}")
            
            # Normalize both passphrases for comparison
            normalized_login = normalize_text(login_text)
            normalized_stored = normalize_text(user_passphrase)
            logger.info(f"Normalized login: {normalized_login}")
            logger.info(f"Normalized stored: {normalized_stored}")
            
            # Check if the recognized text matches the stored passphrase
            passphrase_match = normalized_login == normalized_stored
            logger.info(f"Passphrase match: {passphrase_match}")
            
            # Both voice biometrics AND passphrase content must match
            if similarity >= 0.955 and passphrase_match:  # Adjusted threshold to 0.955
                session['username'] = username
                session['logged_in'] = True
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'redirect': '/quotes'
                })
            else:
                if similarity < 0.955:
                    logger.warning(f"Voice similarity too low: {similarity:.6f}")
                if not passphrase_match:
                    logger.warning(f"Passphrase mismatch: '{normalized_login}' != '{normalized_stored}'")
                return jsonify({
                    'success': False,
                    'error': f'Voice authentication failed. Similarity score: {similarity:.6f}'
                }), 401
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'success': False, 'error': 'An error occurred during login'}), 500
            
    except Exception as e:
        logger.error(f"Login route error: {str(e)}")
        return jsonify({'success': False, 'error': 'An error occurred during login'}), 500

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
        is_partial = data.get('is_partial', False)
        
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
                
                # For partial transcriptions, only return the transcribed text
                if is_partial:
                    return jsonify({
                        'transcribed_text': user_text
                    })
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
    """Get the stored passphrase for a user."""
    try:
        username = request.args.get('username')
        if not username:
            logger.error("No username provided in get_passphrase request")
            return jsonify({"error": "Username is required"}), 400
        
        logger.info(f"Retrieving passphrase for user: {username}")
        
        # Get user details from database
        user_data = get_user(username)
        if not user_data:
            logger.error(f"User not found in database: {username}")
            return jsonify({"error": "User not found"}), 404
            
        # Extract passphrase from user data
        passphrase = user_data[1]  # user_passphrase is at index 1
        if not passphrase:
            logger.error(f"No passphrase found for user: {username}")
            return jsonify({"error": "Passphrase not found"}), 404
            
        logger.info(f"Successfully retrieved passphrase for user: {username}")
        return jsonify({"passphrase": passphrase})
        
    except Exception as e:
        logger.error(f"Error retrieving passphrase: {str(e)}")
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

def verify_password(username, password):
    """Verify if the provided password matches the stored password for the user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
            
        stored_password = result[0]
        return password == stored_password
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False

def get_user_passphrase(username):
    """Retrieve the stored passphrase for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_passphrase FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        return result[0]
    except Exception as e:
        logger.error(f"Error getting user passphrase: {str(e)}")
        return None

def get_user_voice_features(username):
    """Retrieve the stored voice features for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT voice_features FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        return pickle.loads(result[0])
    except Exception as e:
        logger.error(f"Error getting user voice features: {str(e)}")
        return None

def recognize_speech(audio_path):
    """Convert speech in audio file to text using Google Speech Recognition."""
    try:
        # Convert WebM to WAV if needed
        if audio_path.endswith('.webm'):
            wav_path = convert_webm_to_wav(audio_path)
            if not wav_path:
                logger.error("Failed to convert WebM to WAV")
                return None
            audio_path = wav_path
            
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
            
            try:
                # Try with language model
                text = recognizer.recognize_google(audio, language='en-US')
                logger.info(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                logger.error("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                return None
                
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}")
        return None
    finally:
        # Clean up temporary WAV file if it was created
        if audio_path.endswith('.wav') and audio_path != audio_path:
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary WAV file: {str(e)}")

def normalize_text(text):
    """Normalize text for comparison by converting to lowercase and removing extra spaces."""
    if not text:
        return ""
    return ' '.join(text.lower().strip().split())

@app.route('/get_quote', methods=['GET'])
def get_quote():
    try:
        user_text = request.args.get('text', '')
        if not user_text:
            return jsonify({'error': 'No text provided'}), 400
            
        result = get_quote_and_suggestion(user_text)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error getting quote: {str(e)}")
        return jsonify({'error': 'Failed to get quote'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    logger.info(f"Available routes: {[str(rule) for rule in app.url_map.iter_rules()]}")
    app.run(debug=True, host='0.0.0.0', port=5000)