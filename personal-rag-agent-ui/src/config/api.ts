export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
  endpoints: {
    generateResponse: '/generate-response',
    health: '/health',
    files: '/files',
    storageInfo: '/storage/info',
  },
  timeout: 120000, // 2 minutes
};

export const getApiUrl = (endpoint: keyof typeof API_CONFIG.endpoints): string => {
  return `${API_CONFIG.baseUrl}${API_CONFIG.endpoints[endpoint]}`;
};