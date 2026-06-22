import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';

// Pages
import Dashboard from './pages/Dashboard';
import Chunker from './pages/Chunker';
import Index from './pages/Index';
import Query from './pages/Query';
import Benchmark from './pages/Benchmark';

// Create TanStack Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="chunker" element={<Chunker />} />
            <Route path="index" element={<Index />} />
            <Route path="query" element={<Query />} />
            <Route path="benchmark" element={<Benchmark />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </QueryClientProvider>
  );
}

export default App;
