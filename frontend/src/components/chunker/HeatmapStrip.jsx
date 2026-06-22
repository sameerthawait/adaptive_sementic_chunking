import React, { useMemo } from 'react';
import { Card } from '../ui/Card';

export function HeatmapStrip({ perplexityScores, boundaries }) {
  if (!perplexityScores || perplexityScores.length === 0) return null;

  const maxVal = Math.max(...perplexityScores) || 1;
  const minVal = Math.min(...perplexityScores) || 0;
  const range = maxVal - minVal || 1;
  const totalSentences = perplexityScores.length;

  // Helper to interpolate between blue (#2C5F8A) and amber (#B45309)
  const interpolateColor = (val) => {
    const factor = range > 0 ? (val - minVal) / range : 0.5;
    // Low: rgb(44, 95, 138) / #2C5F8A
    // High: rgb(180, 83, 9) / #B45309
    const r = Math.round(44 + factor * (180 - 44));
    const g = Math.round(95 + factor * (83 - 95));
    const b = Math.round(138 + factor * (9 - 138));
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Build chunk spans for the label alignments
  const chunkSpans = useMemo(() => {
    if (!boundaries || boundaries.length < 2) return [];
    
    // Sort boundaries just in case
    const sorted = [...boundaries].sort((a, b) => a - b);
    const spans = [];
    for (let i = 0; i < sorted.length - 1; i++) {
      const start = sorted[i];
      const end = sorted[i + 1];
      spans.push({
        index: i,
        start,
        end,
        sentenceCount: end - start,
      });
    }
    return spans;
  }, [boundaries]);

  return (
    <Card className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">Semantic Map</h3>
        <span className="text-xs text-t2 font-medium">{totalSentences} sentences</span>
      </div>

      <div className="flex flex-col w-full">
        {/* Horizontal strip */}
        <div className="flex w-full h-8 border border-border rounded-[4px] overflow-hidden bg-surface-2 relative shadow-inner">
          {perplexityScores.map((score, idx) => {
            const color = interpolateColor(score);
            const isBoundary = boundaries?.includes(idx);
            
            return (
              <React.Fragment key={`sentence-${idx}`}>
                {isBoundary && idx > 0 && (
                  <div 
                    className="w-[2px] h-full bg-white shrink-0 z-10" 
                    title={`Boundary Split between Sentence ${idx} and ${idx + 1}`}
                  />
                )}
                <div 
                  className="flex-1 h-full cursor-pointer relative group transition-all duration-200 hover:opacity-90"
                  style={{ backgroundColor: color }}
                >
                  {/* Tooltip Card */}
                  <div className="opacity-0 group-hover:opacity-100 pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-t1 text-white text-[10px] py-1 px-2.5 rounded shadow-lg z-50 transition-opacity font-semibold">
                    Sentence {idx + 1}: {score.toFixed(4)} {isBoundary ? ' (⚑ Boundary)' : ''}
                  </div>
                </div>
              </React.Fragment>
            );
          })}
        </div>

        {/* Chunk span labels below the strip */}
        {chunkSpans.length > 0 && (
          <div className="flex w-full mt-1.5 text-[9px] font-bold font-mono divide-x divide-border/20 text-t2 select-none">
            {chunkSpans.map((span, idx) => {
              const percentage = (span.sentenceCount / totalSentences) * 100;
              return (
                <div 
                  key={idx}
                  style={{ width: `${percentage}%` }}
                  className="text-center truncate px-1 text-t2 border-border/10"
                  title={`Chunk ${idx + 1}: Sentences S${span.start + 1} - S${span.end}`}
                >
                  Chunk {idx + 1}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}
