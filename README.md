# Voice Biometric Authentication with Quote Recommendation System

This project combines voice biometric authentication with a sentiment-based quote recommendation system. Users can register and log in using their voice, and once authenticated, they can receive personalized quotes based on their mood or input.
![Screenshot 2025-04-14 170507](https://github.com/user-attachments/assets/7b83b6d3-ec88-4dd2-bef0-8ca35fcc81bf)

![Screenshot 2025-04-14 170521](https://github.com/user-attachments/assets/237e2d2f-8ecc-4247-a167-fe87e7e297f7)

![Screenshot 2025-04-14 170220](https://github.com/user-attachments/assets/3c15cc15-d53e-41b4-a319-45f60138b9a2)

![Screenshot 2025-04-14 170609](https://github.com/user-attachments/assets/c3a033f9-370a-41bc-bfc6-816b449acb2a)

## Features

- **Voice Biometric Authentication**
  - Secure user registration with voice samples
  - Voice-based login verification
  - Passphrase verification with flexible matching

- **Quote Recommendation System**
  - Sentiment analysis of user input
  - Personalized quote recommendations
  - Support for both text and voice input

- **User Management**
  - Secure user registration
  - Session-based authentication
  - Voice feature storage and comparison

## Prerequisites

- Python 3.7+
- MySQL Server
- FFmpeg
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
```bash
# On Windows
choco install ffmpeg

# On Linux
sudo apt-get install ffmpeg

# On Mac
brew install ffmpeg
```

4. Set up MySQL database:
```sql
CREATE DATABASE voice_biometric;
```

5. Configure database connection in `app.py`:
```python
def get_db_connection():
    conn = pymysql.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="voice_biometric"
    )
    return conn
```

## Project Structure

```
.
├── app.py                  # Main application file
├── quotes.json            # Quote database
├── requirements.txt       # Python dependencies
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
│   ├── index.html       # Login/Registration page
│   └── quotes.html      # Quote recommendation page
└── audio_files/         # Directory for audio files
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the application at `http://localhost:5000`

3. Registration Process:
   - Enter username and password
   - Record your voice sample
   - Speak your passphrase
   - Complete registration

4. Login Process:
   - Enter username and password
   - Record your voice sample
   - Get authenticated

5. Quote Recommendation:
   - Enter text or record voice
   - Get sentiment analysis
   - Receive personalized quote

## Technical Details

- **Voice Processing**
  - WebM to WAV conversion using FFmpeg
  - MFCC and pitch feature extraction
  - Cosine similarity for voice comparison

- **Authentication**
  - Voice feature storage in MySQL
  - Session-based authentication
  - Secure password handling

- **Quote Recommendation**
  - Sentiment analysis using TextBlob
  - Tag-based quote matching
  - Random quote selection

## Security Features

- Secure session management
- Voice feature encryption
- Input validation
- Error handling and logging

## Error Handling

The application includes comprehensive error handling for:
- Voice recording issues
- Authentication failures
- Database errors
- File processing errors

## Logging

Detailed logging is implemented for:
- User actions
- Authentication attempts
- Voice processing
- Quote selection
- Error tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask for the web framework
- MySQL for database management
- FFmpeg for audio processing
- TextBlob for sentiment analysis
- SpeechRecognition for voice-to-text conversion # Biometric_Speech_recogniton

