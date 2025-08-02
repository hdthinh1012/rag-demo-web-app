# Demo Instructions

## Quick Demo of Personal RAG Agent UI

### Prerequisites
1. **Backend Running**: Make sure your Flask backend is running on `http://localhost:5000`
2. **Frontend Running**: Start this Next.js app with `npm run dev`

### Demo Flow

#### 1. **Start the Application**
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000)

#### 2. **Basic Chat (Without Files)**
- Type a question like: "What documents do you have indexed?"
- Click Send or press Enter
- You should see a response based on existing documents in your backend

#### 3. **File Upload Demo**
- **Drag & Drop**: Drag a PDF file onto the upload area
- **Or Click**: Click the upload area to browse for files
- **Multiple Files**: Try uploading multiple files at once
- **File Management**: Remove files using the X button before sending

#### 4. **Chat with Uploaded Files**
- Upload a PDF document
- Ask questions like:
  - "Summarize this document"
  - "What are the main topics discussed?"
  - "Extract key findings from the document"

#### 5. **Test Error Handling**
- Try sending a message with the backend stopped
- Upload an unsupported file type
- Test with very long messages

### Expected Features

✅ **Chat Interface**
- Clean, modern chat bubbles
- User messages on the right (blue)
- Assistant messages on the left (white)
- Timestamps on all messages

✅ **File Upload**
- Drag and drop functionality
- File preview with size information
- Remove files before sending
- Support for PDF and TXT files

✅ **Real-time Feedback**
- Loading spinner while processing
- Error messages for failures
- Auto-scroll to latest messages

✅ **Responsive Design**
- Works on desktop and mobile
- Auto-resizing text input
- Proper touch interactions

### Troubleshooting

**Backend Connection Issues:**
- Check if Flask backend is running on port 5000
- Look for CORS errors in browser console
- Verify the API endpoint in browser dev tools

**File Upload Issues:**
- Check browser console for errors
- Ensure files are PDF or TXT format
- Verify backend file size limits

**UI Issues:**
- Refresh the page
- Check browser console for JavaScript errors
- Ensure all dependencies are installed (`npm install`)

### Advanced Testing

1. **Multiple File Upload**: Try uploading 3-4 documents and asking comparative questions
2. **Long Conversations**: Have an extended chat to test scrolling and memory
3. **Mixed Content**: Upload files and ask follow-up questions without files
4. **Error Recovery**: Test error scenarios and recovery

### Performance Notes

- Initial file processing may take time depending on document size
- Large documents will have longer response times
- Multiple files increase processing time

Enjoy testing your Personal RAG Agent! 🤖