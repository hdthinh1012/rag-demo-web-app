import io
import wave
from flask import Flask, request, jsonify, Response
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

from llm.rag_utils import rag, build_index
from llm.config import client, MODEL, TEXT_EMBEDDING_MODEL

app = Flask(__name__)

# Create a temporary directory for uploaded files
os.makedirs("temp", exist_ok=True)


@app.route('/generate-speech', methods=['POST'])
async def generate_speech():
    """
    API endpoint to generate speech from text using a RAG model.
    Expects a POST request with form-data containing:
    - text: The text to be converted to speech.
    - language: The target language for the speech.
    - documents: The document files for RAG.
    """
    print("0")
    print(request.files)
    print(request.form)
    if 'documents' not in request.files:
        return jsonify({"error": "No documents part in the request"}), 400

    print("1")
    text = request.form.get('text')
    language = request.form.get('language')  # Language is not used yet in RAG, but kept for future
    files = request.files.getlist('documents')

    if not text or not language:
        return jsonify({"error": "Missing 'text' or 'language' in form data"}), 400
    print("2")
    # 1. Save uploaded documents
    document_paths = []
    for file in files:
        if file.filename:
            filepath = os.path.join("temp", file.filename)
            file.save(filepath)
            document_paths.append(filepath)

    # 2. Run the RAG pipeline
    try:
        print("3")
        # First, build the index from the documents
        vector_db = build_index(
            document_paths=document_paths,
            embedding_client=client,
            embedding_model=TEXT_EMBEDDING_MODEL,
        )

        audio_data = await rag(
            question=text,
            vector_db=vector_db,
            embedding_client=client,
            embedding_model=TEXT_EMBEDDING_MODEL,
            llm_client=client,
            llm_model=MODEL,
            top_k=3,
            modality="audio"
        )


        if audio_data is None:
            return jsonify({"error": "Failed to generate audio. The RAG pipeline returned no data."}), 500

        # 3. Create a WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)
            wf.writeframes(audio_data.tobytes())
        
        # For testing: Save the audio file to the local system
        with open(os.path.join("temp", "output.wav"), "wb") as f_out:
            f_out.write(wav_buffer.getvalue())

        wav_buffer.seek(0)

        return Response(wav_buffer, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": f"An error occurred in the RAG pipeline: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001) 