import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';

export function SourceChunk({ sourceName, text, score, index, metadata = {} }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Format relevance score as percentage if available, fallback to index rank
  const relevanceLabel = score !== undefined 
    ? `Relevance: ${(score * 100).toFixed(0)}%`
    : `Rank #${index + 1}`;

  return (
    <Card className="hover:border-signal/50 hover:shadow-md transition-all duration-200 p-4 flex flex-col gap-3 bg-white border border-border rounded-xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/20 pb-2">
        <div className="flex items-center gap-1.5 text-[10px] text-t2 font-bold uppercase tracking-wider min-w-0">
          <FileText className="w-3.5 h-3.5 text-t3 shrink-0" />
          <span className="truncate">Source: <span className="text-t1 font-sans font-semibold normal-case">{sourceName || 'Unknown File'}</span></span>
        </div>
        <Badge variant={score !== undefined && score > 0.7 ? 'success' : 'info'}>
          {relevanceLabel}
        </Badge>
      </div>

      {/* Excerpt (expandable) */}
      <div className="flex flex-col gap-1.5">
        <div 
          onClick={() => setIsExpanded(!isExpanded)}
          className={`text-xs text-t2 leading-relaxed bg-surface-2/30 p-2.5 rounded-lg border border-border/10 font-sans cursor-pointer hover:bg-surface-2/50 transition-all ${
            isExpanded ? '' : 'line-clamp-3'
          }`}
        >
          {text}
        </div>
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-[10px] font-bold text-signal hover:underline flex items-center gap-0.5 select-none self-start"
        >
          {isExpanded ? (
            <>
              Show less <ChevronUp className="w-3 h-3" />
            </>
          ) : (
            <>
              Show more <ChevronDown className="w-3 h-3" />
            </>
          )}
        </button>
      </div>

      {/* Metadata Footer */}
      {(metadata.chunk_index !== undefined || metadata.sentence_start !== undefined) && (
        <div className="flex items-center gap-4 text-[9px] text-t3 font-bold uppercase tracking-wider select-none border-t border-border/10 pt-2">
          {metadata.chunk_index !== undefined && (
            <span>Chunk Index: <span className="font-mono text-t2">{metadata.chunk_index}</span></span>
          )}
          {metadata.sentence_start !== undefined && (
            <span>Span: <span className="font-mono text-t2">S{metadata.sentence_start + 1} – S{metadata.sentence_end}</span></span>
          )}
        </div>
      )}
    </Card>
  );
}
