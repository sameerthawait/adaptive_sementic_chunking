import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { SlidersHorizontal, ChevronDown } from 'lucide-react';

export function SearchBar({
  query,
  setQuery,
  collections,
  activeCollection,
  setActiveCollection,
  k,
  setK,
  useRag,
  setUseRag,
  onSearch,
  isLoading,
  retrievalMode,
  setRetrievalMode,
  mmrLambda,
  setMmrLambda,
  filters,
  setFilters,
  sourcesList = []
}) {
  const [isOptionsOpen, setIsOptionsOpen] = useState(false);

  const activeFilterCount = [
    filters?.source,
    filters?.minPerplexity,
    filters?.maxPerplexity,
    filters?.minSentences,
    filters?.maxSentences,
    filters?.chunkType
  ].filter(val => val !== undefined && val !== '').length;

  const handleResetFilters = (e) => {
    e.stopPropagation();
    if (setFilters) {
      setFilters({
        source: '',
        minPerplexity: '',
        maxPerplexity: '',
        minSentences: '',
        maxSentences: '',
        chunkType: ''
      });
    }
    if (setRetrievalMode) setRetrievalMode('hybrid');
    if (setMmrLambda) setMmrLambda(0.5);
  };

  return (
    <Card className="flex flex-col gap-4">
      {/* Top search bar row */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your indexed documents..."
          className="flex-1 p-3 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal"
          onKeyDown={(e) => e.key === 'Enter' && query.trim() && !isLoading && onSearch()}
          disabled={isLoading}
        />
        <Button 
          onClick={() => onSearch()} 
          disabled={isLoading || !query.trim() || !activeCollection}
          className="shrink-0 text-xs"
        >
          {isLoading ? 'Searching...' : 'Send'}
        </Button>
      </div>

      {/* Main settings row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 border-t border-border/30 pt-4">
        {/* Collection Dropdown */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Active Collection</label>
          <select
            value={activeCollection}
            onChange={(e) => setActiveCollection(e.target.value)}
            className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
            disabled={isLoading}
          >
            {!Array.isArray(collections) || collections.length === 0 ? (
              <option value="">No collections available</option>
            ) : (
              collections.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Retrieval Size k */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Retrieval Limit (k)</label>
          <input
            type="number"
            value={k}
            onChange={(e) => setK(parseInt(e.target.value) || 4)}
            min="1"
            max="20"
            className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
            disabled={isLoading}
          />
        </div>

        {/* Enable Agentic RAG Toggle */}
        <div className="flex items-center gap-2 select-none pt-4 sm:pt-2">
          <input
            type="checkbox"
            id="use_rag_checkbox"
            checked={useRag}
            onChange={(e) => setUseRag(e.target.checked)}
            className="w-4 h-4 rounded text-signal border-border focus:ring-signal"
            disabled={isLoading}
          />
          <label htmlFor="use_rag_checkbox" className="text-[10px] font-bold text-t2 uppercase tracking-wider cursor-pointer">
            Enable Agentic RAG
          </label>
        </div>
      </div>

      {/* Advanced collapsible trigger */}
      <div className="border-t border-border/20 pt-3">
        <button
          type="button"
          onClick={() => setIsOptionsOpen(!isOptionsOpen)}
          className="flex items-center justify-between w-full py-1 text-xs font-bold text-t2 uppercase tracking-wider hover:text-signal transition-colors group"
        >
          <span className="flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-t3 group-hover:text-signal transition-colors" />
            Advanced Search Options
            {activeFilterCount > 0 && (
              <span className="bg-signal text-white px-2 py-0.5 rounded-full text-[9px] font-extrabold normal-case leading-none">
                {activeFilterCount} active
              </span>
            )}
          </span>
          <span className="flex items-center gap-2">
            {activeFilterCount > 0 && (
              <span 
                onClick={handleResetFilters}
                className="text-[10px] text-t3 hover:text-boundary normal-case font-medium tracking-normal transition-colors"
              >
                Clear all
              </span>
            )}
            <ChevronDown className={`w-4 h-4 text-t3 transition-transform duration-200 ${isOptionsOpen ? 'rotate-180' : ''}`} />
          </span>
        </button>

        {isOptionsOpen && (
          <div className="mt-4 pt-4 border-t border-border/35 grid grid-cols-1 md:grid-cols-2 gap-6 animate-fadeIn">
            {/* Left Column: Retrieval Mode & MMR */}
            <div className="flex flex-col gap-4">
              {/* Retrieval Mode */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Retrieval Mode</label>
                <div className="flex gap-2">
                  {['hybrid', 'vector', 'keyword'].map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setRetrievalMode(mode)}
                      className={`flex-1 px-3 py-2 rounded-lg text-xs font-bold transition-all border ${
                        retrievalMode === mode
                          ? 'bg-signal text-white border-transparent shadow-sm'
                          : 'bg-surface-2 hover:bg-border/20 text-t2 border-border/50'
                      }`}
                    >
                      {mode === 'hybrid' ? 'Hybrid' : mode === 'vector' ? 'Vector Only' : 'Keyword Only'}
                    </button>
                  ))}
                </div>
                <span className="text-[10px] text-t3 leading-normal">
                  {retrievalMode === 'hybrid' && 'Reciprocal Rank Fusion (RRF) combines semantic vector match and keyword queries.'}
                  {retrievalMode === 'vector' && 'Pure vector search matching chunk embeddings via cosine similarity.'}
                  {retrievalMode === 'keyword' && 'Pure BM25 keyword matching for exact terms.'}
                </span>
              </div>

              {/* MMR Diversity Slider */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">MMR Diversity (λ)</label>
                  <span className="font-mono text-xs font-bold text-signal">{mmrLambda.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={mmrLambda}
                  onChange={(e) => setMmrLambda(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-surface-2 rounded-lg appearance-none cursor-pointer accent-signal focus:outline-none"
                />
                <div className="flex justify-between text-[9px] text-t3 uppercase font-bold tracking-wider">
                  <span>Diverse (0.0)</span>
                  <span>Balanced (0.5)</span>
                  <span>Relevant (1.0)</span>
                </div>
              </div>
            </div>

            {/* Right Column: Metadata Filters */}
            <div className="flex flex-col gap-4 border-l border-border/30 pl-0 md:pl-6">
              <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Metadata Filters</label>

              <div className="grid grid-cols-2 gap-3">
                {/* Source selector */}
                <div className="flex flex-col gap-1 col-span-2">
                  <label className="text-[9px] font-bold text-t3 uppercase">Document Source</label>
                  <select
                    value={filters?.source || ''}
                    onChange={(e) => setFilters && setFilters(prev => ({ ...prev, source: e.target.value }))}
                    className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
                  >
                    <option value="">All Sources</option>
                    {Array.isArray(sourcesList) && sourcesList.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                {/* Perplexity range */}
                <div className="flex flex-col gap-1">
                  <label className="text-[9px] font-bold text-t3 uppercase">Min Perplexity</label>
                  <input
                    type="number"
                    placeholder="0.0"
                    value={filters?.minPerplexity || ''}
                    onChange={(e) => setFilters && setFilters(prev => ({ ...prev, minPerplexity: e.target.value }))}
                    className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-[9px] font-bold text-t3 uppercase">Max Perplexity</label>
                  <input
                    type="number"
                    placeholder="100.0"
                    value={filters?.maxPerplexity || ''}
                    onChange={(e) => setFilters && setFilters(prev => ({ ...prev, maxPerplexity: e.target.value }))}
                    className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
                  />
                </div>

                {/* Sentence count limits */}
                <div className="flex flex-col gap-1">
                  <label className="text-[9px] font-bold text-t3 uppercase">Min Sentences</label>
                  <input
                    type="number"
                    placeholder="1"
                    value={filters?.minSentences || ''}
                    onChange={(e) => setFilters && setFilters(prev => ({ ...prev, minSentences: e.target.value }))}
                    className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-[9px] font-bold text-t3 uppercase">Max Sentences</label>
                  <input
                    type="number"
                    placeholder="100"
                    value={filters?.maxSentences || ''}
                    onChange={(e) => setFilters && setFilters(prev => ({ ...prev, maxSentences: e.target.value }))}
                    className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
                  />
                </div>

                {/* Chunk Type select */}
                <div className="flex flex-col gap-1 col-span-2">
                  <label className="text-[9px] font-bold text-t3 uppercase">Chunk Type</label>
                  <select
                    value={filters?.chunkType || ''}
                    onChange={(e) => setFilters && setFilters(prev => ({ ...prev, chunkType: e.target.value }))}
                    className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
                  >
                    <option value="">All Chunk Types</option>
                    <option value="standard">Standard (Body)</option>
                    <option value="boundary">Boundary / Discontinuous</option>
                    <option value="header">Header</option>
                    <option value="list">List Item</option>
                    <option value="table">Table</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
