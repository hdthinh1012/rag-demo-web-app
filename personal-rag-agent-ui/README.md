# Personal RAG Agent UI

A modern, responsive chatbot interface built with Next.js for interacting with your Personal RAG (Retrieval-Augmented Generation) Agent backend.

## Features

- 🤖 **Interactive Chat Interface** - Clean, modern chat UI with message bubbles
- 📁 **File Upload Support** - Drag & drop or click to upload PDF and TXT files
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- ⚡ **Real-time Updates** - Live chat with loading states and error handling
- 🎨 **Beautiful UI** - Built with Tailwind CSS and Lucide React icons
- 🔄 **Persistent Chat History** - Messages persist during the session
- 📊 **File Management** - View and remove uploaded files before sending

## Prerequisites

- Node.js 18+ installed on your system
- Your Personal RAG Agent Flask backend running (typically on `http://localhost:5000`)

## Quick Start

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Configure API Endpoint** (Optional)
   Create a `.env.local` file in the root directory:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:5000
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

4. **Open Your Browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Usage

### Basic Chat
1. Type your question in the chat input box
2. Click **Send** or press **Enter**
3. Wait for the AI response based on your documents

### File Upload
1. **Drag & Drop**: Drag files directly onto the upload area
2. **Click Upload**: Click the upload area to browse and select files
3. **Supported Formats**: PDF (.pdf) and Text (.txt) files
4. **Multiple Files**: Upload multiple files at once
5. **File Management**: Remove files before sending using the X button

### Chat Features
- **Auto-scroll**: Automatically scrolls to latest messages
- **Timestamps**: Each message shows when it was sent
- **Loading States**: Visual feedback while processing
- **Error Handling**: Clear error messages if something goes wrong
- **Responsive Input**: Text area auto-resizes as you type

## Project Structure

```
personal-rag-agent-ui/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Main page
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   └── ChatInterface.tsx   # Main chat component
│   ├── config/
│   │   └── api.ts              # API configuration
│   └── types/
│       └── chat.ts             # TypeScript types
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── README.md
```

## API Integration

The UI communicates with your Flask backend through these endpoints:

- `POST /generate-response` - Send messages and files for processing
- `GET /health` - Check backend status
- `GET /files` - List uploaded files
- `GET /storage/info` - Get storage information

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Flask backend URL | `http://localhost:5000` |

### API Configuration

Edit `src/config/api.ts` to modify:
- Base URL for the API
- Endpoint paths
- Request timeout settings

## Development

### Available Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint

# Type checking
npm run type-check
```

### Adding New Features

1. **New Components**: Add to `src/components/`
2. **API Endpoints**: Update `src/config/api.ts`
3. **Types**: Define in `src/types/`
4. **Styling**: Use Tailwind CSS classes

## Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure your Flask backend is running
   - Check the API URL in your environment variables
   - Verify CORS settings in your Flask app

2. **File Upload Issues**
   - Check file size limits in your backend
   - Ensure supported file formats (PDF, TXT)
   - Verify backend file handling endpoints

3. **Build Errors**
   - Run `npm install` to ensure all dependencies are installed
   - Check TypeScript errors with `npm run type-check`
   - Verify Node.js version (18+ required)

### Network Configuration

If running on different hosts or ports:

1. Update `.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://your-backend-host:port
   ```

2. Ensure your Flask backend allows CORS from the frontend URL

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **File Upload**: React Dropzone
- **HTTP Client**: Fetch API

## Backend Integration

This UI is designed to work with the Personal RAG Agent Flask backend. Ensure your backend includes:

- CORS configuration for the frontend URL
- The `/generate-response` endpoint accepting form data
- File upload handling for PDF and TXT files
- Proper error responses in JSON format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Personal RAG Agent system. See the main project for licensing information.

---

For more information about the complete RAG system, see the main project documentation.