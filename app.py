import logging
import os
import uuid
import requests
import whisper
from flask import Flask, request, jsonify
from functools import wraps
from werkzeug.exceptions import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable, TranscriptsDisabled
import re
import random
from datetime import datetime, timedelta
import threading
import time
import json

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Ensure Flask uses UTF-8 encoding
app.config['JSON_AS_ASCII'] = False

class ProxyManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.proxies = []
        self.failed_proxies = set()
        self.last_update = None
        self.update_interval = timedelta(hours=1)
        self.lock = threading.Lock()

    def update_proxies(self):
        """Fetch fresh proxy list from Webshare API"""
        try:
            headers = {'Authorization': f'Token {self.api_key}'}
            response = requests.get(
                'https://proxy.webshare.io/api/proxy/list/',
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            with self.lock:
                self.proxies = [{
                    'username': proxy['username'],
                    'password': proxy['password'],
                    'host': proxy['proxy_address'],
                    'port': proxy['ports']['http'],
                    'id': f"{proxy['proxy_address']}:{proxy['ports']['http']}"
                } for proxy in response.json()['results']]
                self.last_update = datetime.now()
                self.failed_proxies.clear()

            logger.info(f"Successfully updated proxy list. Total proxies: {len(self.proxies)}")
        except Exception as e:
            logger.error(f"Failed to update proxy list: {str(e)}")

    def get_random_proxy(self, exclude_failed=True):
        """Get a random proxy from the list, optionally excluding failed ones"""
        if not self.proxies or (self.last_update and datetime.now() - self.last_update > self.update_interval):
            logger.info("Updating proxy list due to empty list or expired cache")
            self.update_proxies()

        with self.lock:
            available_proxies = [
                p for p in self.proxies
                if not exclude_failed or p['id'] not in self.failed_proxies
            ]

            if not available_proxies:
                if exclude_failed and self.proxies:
                    logger.warning("All proxies have failed, resetting failed list and trying again")
                    self.failed_proxies.clear()
                    return self.get_random_proxy(exclude_failed=False)
                logger.error("No proxies available")
                return None

            proxy = random.choice(available_proxies)
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
            logger.info(f"Selected proxy: {proxy['host']}:{proxy['port']}")

            return {
                'http': proxy_url,
                'https': proxy_url,
                'id': proxy['id']
            }

    def mark_proxy_failed(self, proxy_id):
        """Mark a proxy as failed"""
        with self.lock:
            self.failed_proxies.add(proxy_id)
            logger.warning(f"Marked proxy {proxy_id} as failed. Total failed proxies: {len(self.failed_proxies)}")

def proxy_update_worker(proxy_manager):
    """Background worker to update proxy list periodically"""
    while True:
        logger.info("Running scheduled proxy list update")
        proxy_manager.update_proxies()
        time.sleep(3600)

# Initialize proxy manager
proxy_manager = ProxyManager(os.environ.get('WEBSHARE_API_KEY'))

# Start proxy update worker thread
update_thread = threading.Thread(target=proxy_update_worker, args=(proxy_manager,), daemon=True)
update_thread.start()

# Pre-load the Whisper model
try:
    logger.info("Loading Whisper model...")
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    raise

def authenticate(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.warning("Request received without API key")
            return jsonify({"message": "API key is required"}), 401
        if api_key != os.environ.get('API_KEY'):
            logger.warning("Request received with invalid API key")
            return jsonify({"message": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?]*)',
        r'youtube\.com/embed/([^&\n?]*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def sanitize_text(text):
    """Sanitize text to ensure it's properly encoded and JSON-safe"""
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Ensure the text is valid UTF-8
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        # If there are encoding issues, replace problematic characters
        return text.encode('utf-8', errors='replace').decode('utf-8')

def get_transcript_with_retries(video_id, preferred_lang=None, max_retries=5):
    """Get transcript with proxy rotation and retries"""
    errors = []
    used_proxies = set()

    for attempt in range(max_retries):
        try:
            proxy = proxy_manager.get_random_proxy(exclude_failed=True)
            if not proxy:
                logger.error("No proxies available")
                return None, "No proxies available"

            proxy_id = proxy['id']
            if proxy_id in used_proxies:
                logger.warning(f"Proxy {proxy_id} already used in this attempt, but no other proxies available")
            used_proxies.add(proxy_id)

            logger.info(f"Attempt {attempt + 1}/{max_retries} for video {video_id} using proxy {proxy_id}")

            # Get all available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxy)

            try:
                # First try to get preferred language if specified
                if preferred_lang:
                    try:
                        transcript = transcript_list.find_transcript([preferred_lang])
                        logger.info(f"Found transcript in preferred language: {preferred_lang}")
                        return transcript, None
                    except NoTranscriptFound:
                        logger.info(f"No transcript found in preferred language: {preferred_lang}")
                        pass

                # Try to get English transcript
                try:
                    transcript = transcript_list.find_transcript(['en'])
                    logger.info("Found English transcript")
                    return transcript, None
                except NoTranscriptFound:
                    logger.info("No English transcript found")
                    pass

                # Fall back to the default transcript
                transcript = transcript_list.find_transcript([])
                logger.info(f"Found transcript in default language: {transcript.language_code}")
                return transcript, None

            except NoTranscriptFound:
                return None, "No transcript available for this video"

        except (VideoUnavailable, TranscriptsDisabled) as e:
            logger.error(f"Video error: {str(e)}")
            return None, str(e)
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed with proxy {proxy_id}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            proxy_manager.mark_proxy_failed(proxy_id)
            continue

    return None, f"All retry attempts failed: {'; '.join(errors)}"

@app.route('/transcribe-media', methods=['POST'])
@authenticate
def transcribe_media():
    try:
        data = request.get_json()
        if not data or 'media_url' not in data:
            logger.warning("Request received without media_url")
            return jsonify({"message": "media_url is required"}), 400

        media_url = data['media_url']
        job_id = str(uuid.uuid4())
        temp_file_path = f"/tmp/{job_id}"

        logger.info(f"Starting media download for job {job_id}")

        # Download the media file with timeout
        response = requests.get(media_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Media downloaded successfully for job {job_id}")

        # Transcribe the media file
        logger.info(f"Starting transcription for job {job_id}")
        result = model.transcribe(temp_file_path)
        logger.info(f"Transcription completed for job {job_id}")

        # Clean up
        try:
            os.remove(temp_file_path)
            logger.info(f"Temporary file cleaned up for job {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary file: {str(e)}")

        # Sanitize the transcribed text
        sanitized_text = sanitize_text(result["text"])
        
        return jsonify({"response": sanitized_text}), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading media: {str(e)}")
        return jsonify({"message": f"Error downloading media: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"message": "An error occurred processing your request"}), 500

@app.route('/transcribe-YT', methods=['POST'])
@authenticate
def transcribe_yt():
    try:
        data = request.get_json()
        if not data or 'YT_url' not in data:
            logger.warning("Request received without YT_url")
            return jsonify({"message": "YT_url is required"}), 400

        video_id = extract_video_id(data['YT_url'])
        if not video_id:
            logger.warning("Invalid YouTube URL provided")
            return jsonify({"message": "Invalid YouTube URL"}), 400

        preferred_lang = data.get('language')
        logger.info(f"Starting YouTube transcription for video {video_id}")

        transcript, error = get_transcript_with_retries(video_id, preferred_lang)

        if error:
            return jsonify({"message": error}), 404

        # Fetch the transcript data
        transcript_data = transcript.fetch()
        
        # Sanitize all text in the transcript
        sanitized_transcript = []
        for entry in transcript_data:
            sanitized_entry = {
                'text': sanitize_text(entry.get('text', '')),
                'start': entry.get('start', 0),
                'duration': entry.get('duration', 0)
            }
            sanitized_transcript.append(sanitized_entry)

        response_data = {
            "response": sanitized_transcript,
            "metadata": {
                "language": transcript.language_code,
                "requested_language": preferred_lang
            }
        }

        # Log the response for debugging (first few entries only)
        logger.info(f"Transcript response prepared: {len(sanitized_transcript)} entries, language: {transcript.language_code}")
        if sanitized_transcript:
            logger.info(f"Sample entry: {sanitized_transcript[0]}")

        # Use Flask's jsonify but ensure proper encoding
        response = app.response_class(
            response=json.dumps(response_data, ensure_ascii=False, indent=2),
            status=200,
            mimetype='application/json; charset=utf-8'
        )
        return response

    except Exception as e:
        logger.error(f"Error processing YouTube transcript request: {str(e)}")
        return jsonify({"message": f"An error occurred processing your request: {str(e)}"}), 500

@app.errorhandler(HTTPException)
def handle_exception(e):
    logger.error(f"HTTP exception occurred: {str(e)}")
    return jsonify({"message": str(e)}), e.code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
