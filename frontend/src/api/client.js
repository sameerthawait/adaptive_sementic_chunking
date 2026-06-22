import axios from 'axios';

// Axios instance
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api',
  timeout: 300000,
});

// Response interceptor: normalize all errors to user-friendly messages
// On network error: "Cannot reach API — is it running on port 8000?"
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (!error.response) {
      return Promise.reject(new Error("Cannot reach API — is it running on port 8000?"));
    }
    const message = error.response?.data?.detail || error.message || 'API Request failed';
    return Promise.reject(new Error(message));
  }
);

export const chunkText = (text, options) => api.post('chunk', { text, ...options });
export const indexDocuments = (texts, sources, collection) => api.post('index', { texts, sources, collection });

export const queryRAG = (query, collection, k, options = {}) => {
  const { useHybrid, mmrLambda, filters, vectorWeight, bm25Weight } = options;
  const payload = {
    query,
    collection,
    k,
    use_hybrid: useHybrid !== undefined ? useHybrid : true,
    mmr_lambda: mmrLambda !== undefined ? mmrLambda : 0.5,
    vector_weight: vectorWeight !== undefined ? parseFloat(vectorWeight) : undefined,
    bm25_weight: bm25Weight !== undefined ? parseFloat(bm25Weight) : undefined,
  };
  if (filters) {
    payload.filters = {
      source: filters.source || undefined,
      sources: filters.sources || undefined,
      min_perplexity: filters.minPerplexity !== undefined && filters.minPerplexity !== '' ? parseFloat(filters.minPerplexity) : undefined,
      max_perplexity: filters.maxPerplexity !== undefined && filters.maxPerplexity !== '' ? parseFloat(filters.maxPerplexity) : undefined,
      min_sentences: filters.minSentences !== undefined && filters.minSentences !== '' ? parseInt(filters.minSentences, 10) : undefined,
      max_sentences: filters.maxSentences !== undefined && filters.maxSentences !== '' ? parseInt(filters.maxSentences, 10) : undefined,
      chunk_type: filters.chunkType || undefined,
    };
  }
  return api.post('rag', payload);
};

export const queryRetrieve = (query, collection, k, options = {}) => {
  const { useHybrid, mmrLambda, filters, vectorWeight, bm25Weight } = options;
  const payload = {
    query,
    collection,
    k,
    use_hybrid: useHybrid !== undefined ? useHybrid : true,
    mmr_lambda: mmrLambda !== undefined ? mmrLambda : 0.5,
    vector_weight: vectorWeight !== undefined ? parseFloat(vectorWeight) : undefined,
    bm25_weight: bm25Weight !== undefined ? parseFloat(bm25Weight) : undefined,
  };
  if (filters) {
    payload.filters = {
      source: filters.source || undefined,
      sources: filters.sources || undefined,
      min_perplexity: filters.minPerplexity !== undefined && filters.minPerplexity !== '' ? parseFloat(filters.minPerplexity) : undefined,
      max_perplexity: filters.maxPerplexity !== undefined && filters.maxPerplexity !== '' ? parseFloat(filters.maxPerplexity) : undefined,
      min_sentences: filters.minSentences !== undefined && filters.minSentences !== '' ? parseInt(filters.minSentences, 10) : undefined,
      max_sentences: filters.maxSentences !== undefined && filters.maxSentences !== '' ? parseInt(filters.maxSentences, 10) : undefined,
      chunk_type: filters.chunkType || undefined,
    };
  }
  return api.post('retrieve', payload);
};

export const getHealth = () => api.get('health');
export const getCollections = () => api.get('collections');
export const getSources = (collection = 'default') => api.get(`sources?collection=${collection}`);

// Grouped exports for hook compatibility
export const ascApi = {
  getHealth,
  getCollections,
  getSources,
  chunkText: (text, source = 'api_input', visualize = true, options = {}) => 
    chunkText(text, { source, visualize, ...options }),
  indexTexts: indexDocuments,
  queryRag: (query, collection = 'default', k = 4, useRag = true, options = {}) => 
    queryRAG(query, collection, k, options),
  queryRetrieve: (query, collection = 'default', k = 4, options = {}) =>
    queryRetrieve(query, collection, k, options),
};

export default api;

