import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def init_vertex_ai():
    vertexai.init(project="conversationalai-436500", location="us-central1")
    return GenerativeModel("gemini-1.5-pro-001")

def get_safety_settings():
    return [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
    ]

def analyze_audio(file_path):
    try:
        model = init_vertex_ai()
        
        # Determine MIME type based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.mpeg': 'audio/mpeg'
        }
        mime_type = mime_types.get(file_ext, 'audio/mpeg')
        
        # Create audio part using Part.from_data()
        with open(file_path, 'rb') as audio_file:
            audio_part = Part.from_data(
                data=audio_file.read(), 
                mime_type=mime_type
            )
        
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
        
        response = model.generate_content(
            [audio_part, "Provide a detailed transcript of the audio and perform a comprehensive sentiment analysis."],
            generation_config=generation_config,
            safety_settings=get_safety_settings()
        )
        
        return response.text
    
    except Exception as e:
        return f"Error analyzing audio: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Ensure unique filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            analysis = analyze_audio(filepath)
            
            # Remove file after processing
            os.remove(filepath)
            
            return jsonify({"analysis": analysis})
        
        except Exception as e:
            # Remove file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
