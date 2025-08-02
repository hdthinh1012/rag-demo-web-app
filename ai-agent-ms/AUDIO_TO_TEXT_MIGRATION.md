# Audio to Text Migration Guide

## Overview

Your Flask RAG application has been successfully migrated from audio generation to text generation to resolve policy violation issues with the Gemini Live API.

## Changes Made

### ✅ **1. New Text Generation Functions**

#### `generate_text_content()` - Core text generation
```python
def generate_text_content(query: str, client: Any, model: str) -> str:
    """Function to generate text response for provided query using Gemini API."""
```

**Features:**
- Uses standard Gemini API (not Live API)
- Configurable parameters (temperature, max_tokens, etc.)
- Robust error handling
- No policy violation issues

#### `generate_answer_with_text()` - RAG text generation
```python
@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
def generate_answer_with_text(query: str, context: str, llm_client: Any, model: str) -> str:
    """Generate text answer using LLM with retry logic for API quota management."""
```

**Features:**
- Enhanced prompting for better responses
- Retry logic with exponential backoff
- Context-aware responses
- Graceful error handling

### ✅ **2. Updated API Endpoint**

#### Before: `/generate-speech`
- **Returns**: Audio file (.wav)
- **Issues**: Policy violations
- **Complex**: Audio processing, file handling

#### After: `/generate-response`
- **Returns**: JSON with text response
- **Benefits**: No policy issues
- **Simple**: Direct text consumption

### ✅ **3. Response Format**

#### New JSON Response Structure
```json
{
  "response": "Based on the context provided, here is the answer...",
  "query": "What is the main topic?",
  "timestamp": 1704067200,
  "context_chunks": 3
}
```

**Benefits:**
- ✅ Immediate text access
- ✅ Metadata included
- ✅ Easy to process programmatically
- ✅ No file download required

## API Usage Examples

### Basic Query
```bash
curl -X POST http://localhost:5000/generate-response \
  -F "query=What are the main topics in the documents?"
```

### Query with File Upload
```bash
curl -X POST http://localhost:5000/generate-response \
  -F "query=Summarize this document" \
  -F "files=@document.pdf"
```

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:5000/generate-response",
    data={"query": "What is this document about?"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Answer: {result['response']}")
else:
    print(f"Error: {response.json()['error']}")
```

## Configuration Options

### Text Generation Parameters
Located in `setup.py` -> `generate_text_content()`:

```python
config=types.GenerateContentConfig(
    temperature=0.7,        # Creativity (0.0-1.0)
    max_output_tokens=2048, # Response length
    top_p=0.9,             # Nucleus sampling
    top_k=40               # Top-k sampling
)
```

### Adjust for Your Needs:
- **More Creative**: Increase `temperature` to 0.8-0.9
- **More Factual**: Decrease `temperature` to 0.1-0.3
- **Longer Responses**: Increase `max_output_tokens`
- **Shorter Responses**: Decrease `max_output_tokens`

## Benefits of Text Generation

### 🚀 **Performance**
- **Faster**: No audio processing overhead
- **Lighter**: No large audio file transfers
- **Scalable**: Better for high-traffic scenarios

### 🛡️ **Reliability**
- **No Policy Issues**: Uses standard Gemini API
- **Better Error Handling**: Clear text error messages
- **Stable**: Less complex processing pipeline

### 🔧 **Development**
- **Easier Testing**: Direct text responses
- **Better Debugging**: Clear error messages
- **Simpler Integration**: JSON responses

### 💰 **Cost Efficiency**
- **Lower API Costs**: Text generation is cheaper than audio
- **Reduced Bandwidth**: Text vs audio file sizes
- **Faster Processing**: Quicker response times

## Backward Compatibility

### What Remains the Same
- ✅ File upload functionality
- ✅ Document processing
- ✅ Vector database system
- ✅ Persistent storage
- ✅ File change detection
- ✅ Throttling mechanism

### What Changed
- ❌ Audio file generation removed
- ❌ `/generate-speech` endpoint replaced
- ✅ `/generate-response` endpoint added
- ✅ JSON responses instead of audio files

## Migration Checklist

### ✅ **For Developers**
- [x] Update API endpoint from `/generate-speech` to `/generate-response`
- [x] Change response handling from file download to JSON parsing
- [x] Update client applications to handle text responses
- [x] Test with various query types

### ✅ **For Users**
- [x] Same file upload process
- [x] Same query format
- [x] Faster responses (text vs audio generation)
- [x] No audio playback needed

## Testing

### Run the Test Suite
```bash
cd ai-agent-ms
python test_text_generation.py
```

### Manual Testing
1. **Start the server**: `python app.py`
2. **Test health**: `curl http://localhost:5000/health`
3. **Test query**: `curl -X POST http://localhost:5000/generate-response -F "query=test"`

## Troubleshooting

### Issue: "Failed to generate text response"
**Solutions:**
- Check Google API credentials
- Verify model access permissions
- Check API quotas

### Issue: Empty responses
**Solutions:**
- Ensure documents are indexed
- Check vector database status
- Verify query is not empty

### Issue: Slow responses
**Solutions:**
- Check document index size
- Optimize chunk retrieval (reduce `top_k`)
- Monitor API rate limits

## Advanced Usage

### Custom Text Generation
```python
from setup import generate_text_content

# Direct text generation
response = generate_text_content(
    "Explain quantum computing", 
    client, 
    model
)
```

### Custom RAG with Text
```python
from setup import generate_answer_with_text

# RAG with custom context
response = generate_answer_with_text(
    query="What is machine learning?",
    context="Machine learning is a subset of AI...",
    llm_client=client,
    model=model
)
```

## Next Steps

1. **Update Client Applications**: Change from audio handling to text processing
2. **Optimize Prompts**: Customize the prompt in `generate_answer_with_text()`
3. **Monitor Performance**: Use `/storage/info` to track system health
4. **Scale as Needed**: Adjust text generation parameters for your use case

## Files Modified

1. **`setup.py`**:
   - Added `generate_text_content()`
   - Added `generate_answer_with_text()`

2. **`app.py`**:
   - Updated imports
   - Changed `/generate-speech` to `/generate-response`
   - Modified response format to JSON
   - Removed audio processing code

3. **New Files**:
   - `test_text_generation.py` - Test suite
   - `AUDIO_TO_TEXT_MIGRATION.md` - This guide

Your RAG system now provides fast, reliable text responses without policy violations! 🎉