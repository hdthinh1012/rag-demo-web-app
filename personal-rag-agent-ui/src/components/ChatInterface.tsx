'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Send, 
  Paperclip, 
  X, 
  FileText, 
  Upload,
  MessageCircle,
  Bot,
  User,
  Loader2
} from 'lucide-react';
import { ChatMessage, ChatResponse, ApiError } from '@/types/chat';
import { getApiUrl } from '@/config/api';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  // File upload configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt']
    },
    multiple: true,
    onDrop: (acceptedFiles) => {
      setUploadedFiles(prev => [...prev, ...acceptedFiles]);
    }
  });

  const removeFile = (fileIndex: number) => {
    setUploadedFiles(prev => prev.filter((_, index) => index !== fileIndex));
  };

  const sendMessage = async () => {
    if (!inputValue.trim() && uploadedFiles.length === 0) return;

    const messageId = Date.now().toString();
    const userMessage: ChatMessage = {
      id: messageId,
      content: inputValue.trim() || 'File(s) uploaded',
      type: 'user',
      timestamp: new Date(),
      files: uploadedFiles.length > 0 ? [...uploadedFiles] : undefined
    };

    // Add user message
    setMessages(prev => [...prev, userMessage]);

    // Add loading message for assistant
    const loadingMessageId = (Date.now() + 1).toString();
    const loadingMessage: ChatMessage = {
      id: loadingMessageId,
      content: '',
      type: 'assistant',
      timestamp: new Date(),
      isLoading: true
    };
    setMessages(prev => [...prev, loadingMessage]);

    // Clear input and files
    const currentInput = inputValue.trim();
    const currentFiles = [...uploadedFiles];
    setInputValue('');
    setUploadedFiles([]);
    setIsLoading(true);

    try {
      // Prepare form data
      const formData = new FormData();
      formData.append('query', currentInput || 'Please analyze the uploaded files');
      
      currentFiles.forEach(file => {
        formData.append('files', file);
      });

      // Send request to Flask backend
      const response = await fetch(getApiUrl('generateResponse'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse | ApiError = await response.json();

      // Remove loading message and add actual response
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessageId));

      if ('error' in data) {
        const errorMessage: ChatMessage = {
          id: (Date.now() + 2).toString(),
          content: `Sorry, I encountered an error: ${data.error}`,
          type: 'assistant',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      } else {
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 2).toString(),
          content: data.response,
          type: 'assistant',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      // Remove loading message and add error message
      console.error(error);
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessageId));
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        content: `Sorry, I couldn't connect to the server. Please make sure the Flask backend is running on ${getApiUrl('generateResponse').split('/generate-response')[0]}`,
        type: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto bg-white">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 shadow-md">
        <div className="flex items-center gap-2">
          <Bot className="w-6 h-6" />
          <h1 className="text-xl font-semibold">Personal RAG Agent</h1>
        </div>
        <p className="text-blue-100 text-sm mt-1">
          Ask questions about your documents or upload new files to get started
        </p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <MessageCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium mb-2">Welcome to Personal RAG Agent</h3>
            <p className="text-sm">
              Upload documents and ask questions to get AI-powered answers based on your content.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-4 shadow-sm ${
                message.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-800 border border-gray-200'
              }`}
            >
              <div className="flex items-start gap-2">
                <div className={`p-1 rounded ${
                  message.type === 'user' ? 'bg-blue-500' : 'bg-gray-100'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-4 h-4" />
                  ) : (
                    <Bot className="w-4 h-4" />
                  )}
                </div>
                <div className="flex-1">
                  {message.isLoading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Thinking...</span>
                    </div>
                  ) : (
                    <>
                      <div className="text-sm font-medium mb-1">
                        {message.type === 'user' ? 'You' : 'Assistant'}
                      </div>
                      <div className="text-sm whitespace-pre-wrap">
                        {message.content}
                      </div>
                      {message.files && message.files.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {message.files.map((file, index) => (
                            <div key={index} className="flex items-center gap-2 text-xs opacity-80">
                              <FileText className="w-3 h-3" />
                              <span>{file.name}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className={`text-xs mt-1 opacity-70`}>
                        {formatTimestamp(message.timestamp)}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white p-4">
        {/* File Upload Area */}
        {uploadedFiles.length > 0 && (
          <div className="mb-3 space-y-2">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-100 rounded-lg p-2">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-gray-600" />
                  <span className="text-sm text-gray-700">{file.name}</span>
                  <span className="text-xs text-gray-500">
                    ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </span>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-500 hover:text-red-500 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* File Drop Zone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-4 mb-3 transition-colors cursor-pointer ${
            isDragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex items-center justify-center gap-2 text-gray-600">
            <Upload className="w-4 h-4" />
            <span className="text-sm">
              {isDragActive
                ? 'Drop files here...'
                : 'Click or drag files here to upload (PDF, TXT)'}
            </span>
          </div>
        </div>

        {/* Message Input */}
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="text-black w-full resize-none rounded-lg border border-gray-300 px-4 py-3 pr-12 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 min-h-[50px] max-h-32"
              rows={1}
              disabled={isLoading}
            />
            <button
              type="button"
              className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Paperclip className="w-4 h-4" />
            </button>
          </div>
          <button
            onClick={sendMessage}
            disabled={isLoading || (!inputValue.trim() && uploadedFiles.length === 0)}
            className="bg-blue-600 text-white rounded-lg px-6 py-3 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2 min-w-[100px] justify-center"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Send</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;