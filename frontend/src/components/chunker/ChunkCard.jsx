import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { motion } from 'framer-motion';
import { Database, Loader2 } from 'lucide-react';

export function ChunkCard({ 
  chunk, 
  index, 
  maxPerplexity = 1, 
  onAddToIndex,
  isIndexing = false
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const metadata = chunk.metadata || {};
  const sentenceCount = metadata.sentence_count || (metadata.sentence_end - metadata.sentence_start) || 0;
  
  const avgPpl = metadata.avg_perplexity || 0;
  const percentage = maxPerplexity > 0 ? (avgPpl / maxPerplexity) * 100 : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.06 }}
    >
      <Card className="hover:border-signal/50 hover:shadow-md transition-all duration-200 flex flex-col gap-3.5 p-4 bg-white relative">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/30 pb-2">
          <div className="flex items-center gap-2">
            <span className="bg-signal-light text-signal border border-signal-border/30 px-2 py-0.5 rounded font-bold text-[10px] uppercase">
              Chunk #{index + 1}
            </span>
            <span className="font-mono text-[10px] text-t3 font-bold">
              S{metadata.sentence_start + 1} – S{metadata.sentence_end}
            </span>
          </div>
          <Badge variant="info">
            {sentenceCount} {sentenceCount === 1 ? 'sentence' : 'sentences'}
          </Badge>
        </div>

        {/* Perplexity Bar */}
        <div className="flex items-center gap-2.5">
          <span className="text-[9px] font-bold text-t3 uppercase tracking-wider shrink-0 w-8">PPL</span>
          <div className="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden border border-border/10">
            <div 
              className="h-full bg-signal rounded-full transition-all duration-500" 
              style={{ width: `${percentage}%` }}
            />
          </div>
          <span className="font-bold font-mono text-[11px] text-t1 tabular-nums text-right w-10">
            {avgPpl.toFixed(2)}
          </span>
        </div>

        {/* Chunk Text (Truncated / Expandable) */}
        <div className="flex flex-col gap-1">
          <div className={`text-xs text-t2 leading-relaxed bg-surface-2/40 p-3 rounded-lg border border-border/20 font-sans whitespace-pre-wrap ${isExpanded ? '' : 'line-clamp-3'}`}>
            {chunk.text}
          </div>
          <button 
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-[10px] font-bold text-signal hover:underline flex items-center gap-1 select-none self-start mt-0.5"
          >
            {isExpanded ? 'Show less ↑' : 'Show more ↓'}
          </button>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-border/20 pt-2.5 mt-0.5">
          <span className="text-[10px] text-t3 font-semibold uppercase tracking-wider">
            Avg Perplexity: <span className="font-bold font-mono text-t1 text-xs tabular-nums">{avgPpl.toFixed(3)}</span>
          </span>
          <button
            type="button"
            disabled={isIndexing}
            onClick={() => onAddToIndex?.(chunk.text)}
            className="text-[9px] font-bold text-signal border border-signal-border hover:bg-signal-light hover:text-signal disabled:opacity-50 px-2.5 py-1 rounded bg-transparent transition-all uppercase tracking-wider flex items-center gap-1 cursor-pointer"
          >
            {isIndexing ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                Indexing...
              </>
            ) : (
              <>
                <Database className="w-3 h-3" />
                Add to Index
              </>
            )}
          </button>
        </div>
      </Card>
    </motion.div>
  );
}
