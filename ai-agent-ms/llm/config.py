# For GenerativeAI
from google import genai

# --- Configuration ---
PROJECT_ID = "supple-kayak-466408-a3"  # @param {type:"string"}
LOCATION = "asia-southeast1"  # @param {type:"string"}

# --- Model Names ---
MODEL_ID = "gemini-live-2.5-flash-preview-native-audio"
MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
TEXT_EMBEDDING_MODEL = "gemini-embedding-001"

# --- GenAI Client ---
# Initialize the client for calling the Gemini API in Vertex AI
try:
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
except Exception as e:
    print(f"Error initializing GenAI client: {e}")
    client = None 