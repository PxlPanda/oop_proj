import axios from 'axios';

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
});

export const uploadTemplate = (file) => {
  const form = new FormData();
  form.append('image', file);
  return api.post('/upload-template/', form);
};


export const getSymbols = (sessionId) =>                    api.get(`/session/${sessionId}/symbols/`);
export const updateSymbol = (sessionId, symbolId, data) =>  api.put(`/session/${sessionId}/symbol/${symbolId}/`, data);
export const reviewSession = (sessionId) =>                 api.post(`/session/${sessionId}/review/`);