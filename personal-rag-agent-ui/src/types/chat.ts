export interface ChatMessage {
  id: string;
  content: string;
  type: 'user' | 'assistant';
  timestamp: Date;
  files?: File[];
  isLoading?: boolean;
}

export interface ChatResponse {
  response: string;
  query: string;
  timestamp: number;
  context_chunks: number;
}

export interface ApiError {
  error: string;
}