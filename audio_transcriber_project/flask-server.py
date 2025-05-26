from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from AudioTranscriber import AudioTranscriber

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/audio-processing', methods=['POST'])
def process_audio():
    # Check if file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['audio_file']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the audio
            transcriber = AudioTranscriber()
            results = transcriber.analyze_audio(filepath)
            
            # Clean up - remove the temporary file
            os.remove(filepath)
            
            # Return the results
            return jsonify({
                "status": "success",
                "results": {
                    "transcription": results["transcript"],
                    "speech_analysis": {
                        "speech_rate": results["speech_rate"],
                        "audio_duration": results["audio_duration"],
                        "word_count": results["word_count"]
                    },
                    "filler_words": results["filler_analysis"],
                    "silent_gaps": {
                        "total_gaps": results["gap_count"],
                        "gaps": results["gap_analysis"]
                    }
                }
            })
            
        except Exception as e:
            # Clean up if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True)