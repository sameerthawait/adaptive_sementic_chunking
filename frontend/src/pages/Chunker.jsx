import React, { useState, useMemo } from 'react';
import { useChunk } from '../hooks/useChunk';
import { PerplexityChart } from '../components/chunker/PerplexityChart';
import { HeatmapStrip } from '../components/chunker/HeatmapStrip';
import { ChunkCard } from '../components/chunker/ChunkCard';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Empty } from '../components/ui/Empty';
import { Toast } from '../components/ui/Toast';
import { logActivity } from '../utils/activity';
import { 
  ChevronDown, 
  ChevronUp, 
  Activity, 
  Loader2, 
  Settings 
} from 'lucide-react';
import { motion, useReducedMotion } from 'framer-motion';

export default function Chunker() {
  const [text, setText] = useState('');
  const [source, setSource] = useState('sample_document.txt');
  const [zScoreThreshold, setZScoreThreshold] = useState(2.0);
  const [contextWindow, setContextWindow] = useState(3);
  const [minChunkSentences, setMinChunkSentences] = useState(3);
  const [sentenceOverlap, setSentenceOverlap] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [toastType, setToastType] = useState('info');
  const [indexingIdx, setIndexingIdx] = useState(null);

  const { chunkMutation, indexMutation } = useChunk();

  const handleChunkSubmit = () => {
    if (!text.trim()) return;

    chunkMutation.mutate(
      { 
        text, 
        source, 
        visualize: true,
        options: {
          z_score_threshold: zScoreThreshold,
          context_window: contextWindow,
          min_chunk_sentences: minChunkSentences,
          sentence_overlap: sentenceOverlap
        }
      },
      {
        onSuccess: (data) => {
          setToastType('success');
          setToastMessage(`Segmented successfully into ${data.total_chunks} chunks!`);
          logActivity('chunk', `Segmented '${source}' into ${data.total_chunks} chunks`, 'success');
        },
        onError: (err) => {
          setToastType('error');
          setToastMessage(err.message || 'Chunking failed');
          logActivity('chunk', `Failed to segment '${source}'`, 'failed');
        },
      }
    );
  };

  const handleAddToIndex = (chunkText, idx) => {
    setIndexingIdx(idx);
    indexMutation.mutate(
      { 
        texts: [chunkText], 
        sources: [source || 'single_chunk.txt'], 
        collection: 'default' 
      },
      {
        onSuccess: () => {
          setToastType('success');
          setToastMessage('Chunk indexed successfully into default collection!');
          setIndexingIdx(null);
          logActivity('index', `Indexed segment from '${source}'`, 'success');
        },
        onError: (err) => {
          setToastType('error');
          setToastMessage(err.message || 'Indexing failed');
          setIndexingIdx(null);
          logActivity('index', `Failed to index segment from '${source}'`, 'failed');
        }
      }
    );
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleChunkSubmit();
    }
  };

  const wordCount = useMemo(() => {
    if (!text.trim()) return 0;
    return text.trim().split(/\s+/).length;
  }, [text]);

  const sentenceCount = useMemo(() => {
    if (!text.trim()) return 0;
    const sentences = text.match(/[^.!?]+(?:[.!?]+|$)/g);
    return sentences ? sentences.length : 0;
  }, [text]);

  const results = chunkMutation.data;
  const isLoading = chunkMutation.isPending;

  const maxPerplexity = useMemo(() => {
    if (!results || !results.chunks || results.chunks.length === 0) return 1;
    return Math.max(...results.chunks.map(c => c.metadata?.avg_perplexity || 0)) || 1;
  }, [results]);

  const shouldReduceMotion = useReducedMotion();

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
          PLAYGROUND
        </span>
        <h2 className="text-xl font-bold text-t1 tracking-heading leading-tight">
          Document Segmenter
        </h2>
        <p className="text-xs text-t2 mt-1">
          Segment unstructured documents semantically using adaptive perplexity threshold transitions.
        </p>
      </div>

      <div className="grid grid-cols-1 min-[900px]:grid-cols-[38%_62%] gap-6 items-start">
        {/* Left panel: Text input & Settings */}
        <div className="flex flex-col gap-4">
          <Card className="p-4 flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-t3 uppercase tracking-wider">Document Identifier</label>
              <input
                type="text"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="w-full p-2.5 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal font-sans font-medium"
                placeholder="sample_document.txt"
                disabled={isLoading}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-t3 uppercase tracking-wider">Document Text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Paste your document here..."
                className="w-full min-h-[280px] p-3 text-xs font-mono border border-border rounded-lg bg-surface focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal resize-y leading-relaxed"
                disabled={isLoading}
              />
              <div className="flex justify-between items-center text-[10px] text-t3 px-1 mt-0.5">
                <div>Words: <span className="font-semibold text-t2 font-mono">{wordCount}</span></div>
                <div>Sentences: <span className="font-semibold text-t2 font-mono">{sentenceCount}</span></div>
              </div>
            </div>

            {/* Advanced Settings Accordion */}
            <div className="border border-border/60 rounded-lg overflow-hidden bg-surface-2/40">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between p-3 text-xs font-bold text-t2 hover:bg-surface-2 transition-all select-none"
              >
                <span className="flex items-center gap-1.5">
                  <Settings className="w-3.5 h-3.5 text-t3" />
                  ADVANCED SETTINGS
                </span>
                {showAdvanced ? <ChevronUp className="w-4 h-4 text-t3" /> : <ChevronDown className="w-4 h-4 text-t3" />}
              </button>
              
              {showAdvanced && (
                <div className="p-4 border-t border-border/40 flex flex-col gap-5 bg-surface">
                  {/* Z-score threshold slider */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between items-center text-[10px] font-bold text-t2 uppercase tracking-wider">
                      <span>Z-Score Threshold</span>
                      <span className="font-mono bg-signal-light text-signal px-1.5 py-0.5 rounded font-bold text-xs">{zScoreThreshold.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="4.0"
                      step="0.1"
                      value={zScoreThreshold}
                      onChange={(e) => setZScoreThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-surface-2 rounded-lg appearance-none cursor-pointer accent-signal"
                    />
                    <div className="flex justify-between text-[9px] text-t3 font-semibold">
                      <span>0.5 (More Cuts)</span>
                      <span>4.0 (Fewer Cuts)</span>
                    </div>
                  </div>

                  {/* Context window size radios */}
                  <div className="flex flex-col gap-2">
                    <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Context Window Size</label>
                    <div className="flex items-center gap-4">
                      {[1, 2, 3, 4, 5].map((val) => (
                        <label key={val} className="flex items-center gap-1.5 text-xs text-t2 cursor-pointer select-none font-semibold">
                          <input
                            type="radio"
                            name="contextWindow"
                            value={val}
                            checked={contextWindow === val}
                            onChange={() => setContextWindow(val)}
                            className="w-4 h-4 text-signal border-border focus:ring-signal focus:ring-offset-0 focus:outline-none"
                          />
                          <span>{val}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Min chunk sentences number input */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Min Chunk Sentences</label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={minChunkSentences}
                      onChange={(e) => setMinChunkSentences(Math.max(1, Math.min(10, parseInt(e.target.value, 10) || 1)))}
                      className="w-full p-2 text-xs border border-border rounded-lg bg-surface-2 focus:bg-white focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal font-mono font-bold"
                    />
                  </div>

                  {/* Sentence overlap toggle switch */}
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col pr-2">
                      <span className="text-[10px] font-bold text-t2 uppercase tracking-wider">Sentence Overlap</span>
                      <span className="text-[9px] text-t3 leading-tight mt-0.5">Prepend overlap context from previous chunks</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setSentenceOverlap(!sentenceOverlap)}
                      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-signal focus:ring-offset-2 ${
                        sentenceOverlap ? 'bg-signal' : 'bg-border'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          sentenceOverlap ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Detect Boundaries trigger button */}
            <Button 
              onClick={handleChunkSubmit} 
              disabled={isLoading || !text.trim()} 
              variant="primary"
              className="w-full py-2.5 text-xs font-bold uppercase tracking-wider"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing perplexity...
                </span>
              ) : (
                'Detect Boundaries'
              )}
            </Button>
          </Card>
        </div>

        {/* Right panel: Results / Empty state */}
        <div className="flex flex-col gap-6">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center min-h-[450px] bg-surface border border-border rounded-xl p-12 text-center select-none shadow-sm">
              <div className="relative flex items-center justify-center mb-4">
                <div className="absolute w-12 h-12 border-4 border-signal/20 rounded-full animate-ping" />
                <Loader2 className="relative w-10 h-10 text-signal animate-spin stroke-[1.5]" />
              </div>
              <h3 className="text-sm font-bold text-t1 mb-1 tracking-heading uppercase">Analyzing Perplexity...</h3>
              <p className="text-xs text-t2 max-w-xs leading-normal">
                Running sliding window LLM perplexity evaluation and segment threshold diagnostics.
              </p>
            </div>
          ) : results ? (
            <div className="flex flex-col gap-6 animate-fadeIn">
              {/* Stacked Section 1: Stats Bar */}
              <Card className="flex items-center justify-between divide-x divide-border/40 py-4 px-6 bg-surface shadow-sm">
                <div className="flex-1 text-center">
                  <span className="text-[10px] font-bold text-t3 uppercase tracking-wider block mb-0.5">Total Chunks</span>
                  <span className="text-2xl font-bold text-signal tracking-tight font-mono">{results.total_chunks}</span>
                </div>
                <div className="flex-1 text-center px-2">
                  <span className="text-[10px] font-bold text-t3 uppercase tracking-wider block mb-0.5">Average Perplexity</span>
                  <span className="text-2xl font-bold text-t1 tracking-tight font-mono">{(results.avg_perplexity || 0).toFixed(3)}</span>
                </div>
                <div className="flex-1 text-center">
                  <span className="text-[10px] font-bold text-t3 uppercase tracking-wider block mb-0.5">Boundaries Detected</span>
                  <span className="text-2xl font-bold text-boundary tracking-tight font-mono">{results.boundaries_detected}</span>
                </div>
              </Card>

              {/* Stacked Section 2: Recharts Signal Chart */}
              {results.perplexity_scores && (
                <PerplexityChart
                  perplexityScores={results.perplexity_scores}
                  boundaries={results.boundaries}
                />
              )}

              {/* Stacked Section 3: Heatmap Strip & Chunk Cards List */}
              <div className="flex flex-col gap-6">
                {results.perplexity_scores && (
                  <HeatmapStrip
                    perplexityScores={results.perplexity_scores}
                    boundaries={results.boundaries}
                  />
                )}

                <div className="flex flex-col gap-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">
                      {results.chunks.length} {results.chunks.length === 1 ? 'Chunk' : 'Chunks'}
                    </h3>
                    <span className="text-xs text-t2 font-medium">
                      Avg Perplexity: <span className="font-bold font-mono text-signal">{results.avg_perplexity.toFixed(3)}</span>
                    </span>
                  </div>
                  {results.chunks.length === 0 ? (
                    <Empty title="No chunks created" description="The text could not be divided." />
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {results.chunks.map((chunk, idx) => (
                        <ChunkCard 
                          key={idx} 
                          chunk={chunk} 
                          index={idx} 
                          maxPerplexity={maxPerplexity}
                          onAddToIndex={(txt) => handleAddToIndex(txt, idx)}
                          isIndexing={indexingIdx === idx}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center min-h-[450px] bg-surface border border-border rounded-xl p-12 text-center select-none shadow-sm">
              <div className="p-4 bg-surface-2 border border-border/30 rounded-full mb-3 shadow-inner">
                <Activity className="w-8 h-8 text-t3 stroke-[1.5]" />
              </div>
              <p className="text-xs text-t2 font-medium max-w-xs leading-normal">
                Paste a document to visualize its semantic structure
              </p>
            </div>
          )}
        </div>
      </div>

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
