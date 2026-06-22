import React, { useEffect, useState, useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { useChunk } from '../hooks/useChunk';
import { useRag } from '../hooks/useRag';
import { SourceChunk } from '../components/query/SourceChunk';
import { SearchBar } from '../components/query/SearchBar';
import { ascApi } from '../api/client';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Empty } from '../components/ui/Empty';
import { Toast } from '../components/ui/Toast';
import { logActivity } from '../utils/activity';
import { 
  Search, 
  MessageSquare, 
  Database, 
  History, 
  Clock 
} from 'lucide-react';
import { motion, useReducedMotion } from 'framer-motion';

export default function Query() {
  const location = useLocation();
  const { collectionsQuery } = useChunk();
  const { queryMutation } = useRag();
  const shouldReduceMotion = useReducedMotion();

  const collections = collectionsQuery.data?.collections || [];

  const [queryText, setQueryText] = useState('');
  const [activeCollection, setActiveCollection] = useState('');
  const [k, setK] = useState(3);
  const [useRagFlag, setUseRagFlag] = useState(true);
  const [toastMessage, setToastMessage] = useState('');
  const [toastType, setToastType] = useState('info');

  // New retrieval enhancement states
  const [retrievalMode, setRetrievalMode] = useState('hybrid');
  const [mmrLambda, setMmrLambda] = useState(0.5);
  const [filters, setFilters] = useState({
    source: '',
    minPerplexity: '',
    maxPerplexity: '',
    minSentences: '',
    maxSentences: '',
    chunkType: ''
  });
  const [sourcesList, setSourcesList] = useState([]);

  // Fetch sources list when active collection changes
  useEffect(() => {
    if (!activeCollection) {
      setSourcesList([]);
      return;
    }
    ascApi.getSources(activeCollection)
      .then((res) => {
        setSourcesList(res || []);
      })
      .catch((err) => {
        console.error("Failed to load sources:", err);
      });
  }, [activeCollection]);

  // Load recent queries from localStorage
  const [recentQueries, setRecentQueries] = useState(() => {
    try {
      const raw = localStorage.getItem('asc_recent_queries');
      if (!raw || raw === 'null') return [];
      const list = JSON.parse(raw);
      return Array.isArray(list) ? list.filter(Boolean) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    if (location.state?.collection) {
      setActiveCollection(location.state.collection);
    } else if (collections.length > 0 && !activeCollection) {
      setActiveCollection(collections[0]);
    }
  }, [collections, location.state, activeCollection]);

  const handleSearchSubmit = (overrideQueryText = null) => {
    const textToSearch = (typeof overrideQueryText === 'string') ? overrideQueryText : queryText;
    if (!textToSearch || !textToSearch.trim() || !activeCollection) return;

    let useHybrid = true;
    let vectorWeight = 0.6;
    let bm25Weight = 0.4;
    if (retrievalMode === 'vector') {
      useHybrid = false;
    } else if (retrievalMode === 'keyword') {
      useHybrid = true;
      vectorWeight = 0.0;
      bm25Weight = 1.0;
    }

    queryMutation.mutate(
      { 
        query: textToSearch, 
        collection: activeCollection, 
        k, 
        useRag: useRagFlag,
        useHybrid,
        mmrLambda,
        filters,
        vectorWeight,
        bm25Weight
      },
      {
        onSuccess: (data) => {
          setToastType('success');
          setToastMessage('Query completed successfully!');
          logActivity('query', `Asked: "${textToSearch.length > 40 ? textToSearch.substring(0, 40) + '...' : textToSearch}"`, 'success');
          
          // Save to recent queries
          const newQuery = {
            text: textToSearch,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          };
          const updated = [
            newQuery,
            ...recentQueries.filter(q => q.text !== textToSearch)
          ].slice(0, 5);
          
          setRecentQueries(updated);
          localStorage.setItem('asc_recent_queries', JSON.stringify(updated));
        },
        onError: (err) => {
          setToastType('error');
          setToastMessage(err.message || 'Query failed');
          logActivity('query', `Query failed: "${textToSearch.length > 45 ? textToSearch.substring(0, 45) + '...' : textToSearch}"`, 'failed');
        },
      }
    );
  };

  const results = queryMutation.data;
  const isLoading = queryMutation.isPending;

  const sources = useMemo(() => {
    if (!results) return [];
    if (useRagFlag) {
      return (results.sources || []).map((s, idx) => ({
        sourceName: s.source,
        text: s.page_content,
        index: idx,
        metadata: { chunk_index: s.chunk_index }
      }));
    } else {
      return (results.documents || []).map((doc, idx) => ({
        sourceName: doc.metadata?.source || 'unknown',
        text: doc.page_content,
        score: results.scores?.[idx],
        index: idx,
        metadata: doc.metadata
      }));
    }
  }, [results, useRagFlag]);

  const handleReRun = (q) => {
    setQueryText(q.text);
    handleSearchSubmit(q.text);
  };

  const isDbEmpty = collections.length === 0;

  // Animation variants
  const pageVariants = {
    initial: { opacity: 0, y: shouldReduceMotion ? 0 : 8 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.2, ease: "easeOut" }
  };

  return (
    <motion.div 
      initial="initial"
      animate="animate"
      variants={pageVariants}
      className="flex flex-col gap-6"
    >
      <div>
        <span className="text-[11px] font-bold text-t3 uppercase tracking-wider block mb-1">
          RESEARCH ASSISTANT
        </span>
        <h2 className="text-xl font-bold text-t1 tracking-heading leading-tight">
          Self-Correcting RAG Chat
        </h2>
        <p className="text-xs text-t2 mt-1">
          Retrieve segment boundaries, re-rank contextual content, and generate agentic reasoning completions.
        </p>
      </div>

      {isDbEmpty ? (
        <Empty 
          title="No active collections found"
          description="Index some documents first, then ask questions."
          icon={<Database className="w-8 h-8 text-t3 animate-pulse" />}
        />
      ) : (
        <div className="flex flex-col gap-6">
          <SearchBar
            query={queryText}
            setQuery={setQueryText}
            collections={collections}
            activeCollection={activeCollection}
            setActiveCollection={setActiveCollection}
            k={k}
            setK={setK}
            useRag={useRagFlag}
            setUseRag={setUseRagFlag}
            onSearch={handleSearchSubmit}
            isLoading={isLoading}
            retrievalMode={retrievalMode}
            setRetrievalMode={setRetrievalMode}
            mmrLambda={mmrLambda}
            setMmrLambda={setMmrLambda}
            filters={filters}
            setFilters={setFilters}
            sourcesList={sourcesList}
          />

          {/* Loading Skeleton View */}
          {isLoading && (
            <motion.div
              animate={shouldReduceMotion ? {} : { opacity: [0.4, 0.8, 0.4] }}
              transition={shouldReduceMotion ? {} : { repeat: Infinity, duration: 1 }}
              className="flex flex-col gap-4 bg-white border border-border rounded-xl p-6 shadow-sm"
            >
              <div className="flex items-center gap-2">
                <div className="h-5 bg-border/40 rounded w-24" />
                <div className="h-5 bg-border/40 rounded w-16" />
              </div>
              <div className="space-y-2 mt-2">
                <div className="h-4 bg-border/30 rounded w-full" />
                <div className="h-4 bg-border/30 rounded w-11/12" />
                <div className="h-4 bg-border/30 rounded w-4/5" />
                <div className="h-4 bg-border/30 rounded w-5/6" />
              </div>
            </motion.div>
          )}

          {/* Results Area */}
          {results && !isLoading && (
            <div className="flex flex-col gap-6">
              {/* Answer card */}
              {useRagFlag && results.answer && (
                <Card className="flex flex-col gap-4 p-6 bg-white border border-border rounded-xl shadow-sm">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/20 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="bg-signal text-white px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border border-transparent">
                        RAG Answer
                      </span>
                      {results.iterations !== undefined && (
                        <span className="bg-surface-2 text-t2 border border-border px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider">
                          {results.iterations} {results.iterations === 1 ? 'iteration' : 'iterations'}
                        </span>
                      )}
                    </div>
                    
                    {/* Confidence score bar */}
                    {results.entailment_score !== undefined && (
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-[10px] font-bold text-t3 uppercase tracking-wider">Confidence</span>
                        <div className="w-24 h-1.5 bg-signal-light border border-signal-border/20 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-signal rounded-full"
                            style={{ width: `${Math.max(0, Math.min(1, results.entailment_score)) * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs font-bold text-t1 tabular-nums w-8 text-right">
                          {(results.entailment_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Answer Text: 16px/1.7 */}
                  <div className="text-base leading-relaxed text-t1 font-sans" style={{ fontSize: '16px', lineHeight: '1.7' }}>
                    {results.answer}
                  </div>
                </Card>
              )}

              {/* Source Chunks (below divider "Sources") */}
              {sources.length > 0 && (
                <div className="border-t border-border/40 pt-6 flex flex-col gap-4">
                  <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">Sources</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {sources.map((s, idx) => (
                      <SourceChunk
                        key={idx}
                        sourceName={s.sourceName}
                        text={s.text}
                        score={s.score}
                        index={idx}
                        metadata={s.metadata}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Empty State placeholder */}
          {!results && !isLoading && (
            <Card className="p-8 flex flex-col items-center justify-center text-center bg-surface border border-border rounded-xl">
              <MessageSquare className="w-8 h-8 text-t3 mb-2 stroke-[1.5]" />
              <h4 className="text-xs font-bold text-t1 uppercase tracking-wider">Awaiting Query</h4>
              <p className="text-xs text-t2 mt-1 max-w-xs">Ask a question above to trigger search retrieval and generation.</p>
            </Card>
          )}

          {/* Recent Queries List */}
          {recentQueries.length > 0 && (
            <div className="flex flex-col gap-3 mt-4 border-t border-border/30 pt-4">
              <h3 className="text-xs font-bold text-t2 uppercase tracking-wider flex items-center gap-1.5">
                <History className="w-4 h-4 text-t3" />
                Recent Queries
              </h3>
              <div className="flex flex-col gap-2">
                {recentQueries.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleReRun(q)}
                    disabled={isLoading}
                    className="flex items-center justify-between text-left p-3 rounded-lg border border-border/40 bg-white hover:border-signal/30 hover:bg-signal-light/10 transition-all text-xs text-t1 group"
                  >
                    <span className="font-medium truncate pr-4 group-hover:text-signal transition-colors">{q.text}</span>
                    <span className="flex items-center gap-1 text-[10px] text-t3 shrink-0 select-none">
                      <Clock className="w-3 h-3" />
                      {q.timestamp}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {toastMessage && (
        <Toast
          message={toastMessage}
          type={toastType}
          onClose={() => setToastMessage('')}
        />
      )}
    </motion.div>
  );
}
