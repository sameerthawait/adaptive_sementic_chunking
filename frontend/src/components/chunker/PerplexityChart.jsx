import React from 'react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ReferenceLine 
} from 'recharts';
import { motion } from 'framer-motion';
import { Badge } from '../ui/Badge';

export function PerplexityChart({ perplexityScores, boundaries }) {
  if (!perplexityScores || perplexityScores.length === 0) return null;

  // Format data for Recharts
  const data = perplexityScores.map((score, idx) => ({
    name: `S${idx + 1}`,
    index: idx,
    score: parseFloat(score.toFixed(4)),
    isBoundary: boundaries?.includes(idx)
  }));

  const boundaryLines = (boundaries || []).filter(idx => idx > 0 && idx < perplexityScores.length);
  const boundariesCount = boundaryLines.length;

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white border border-border rounded-lg p-2.5 shadow-sm text-xs text-t1 font-sans">
          <div className="font-semibold">Sentence {item.index + 1} — Score {item.score.toFixed(4)}</div>
          {item.isBoundary && (
            <div className="text-boundary font-bold flex items-center gap-1 mt-1 font-mono">
              <span>⚑</span> Boundary detected
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }} 
      animate={{ opacity: 1, x: 0 }} 
      transition={{ duration: 0.4 }}
      className="flex flex-col gap-3"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">Perplexity Signal</h3>
        <Badge variant="warning">
          {boundariesCount} {boundariesCount === 1 ? 'boundary' : 'boundaries'} detected
        </Badge>
      </div>

      <div className="h-[200px] w-full bg-white border border-border rounded-[8px] p-3 shadow-sm select-none">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 15, right: 10, left: -25, bottom: 0 }}>
            <defs>
              <linearGradient id="colorSignal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--signal)" stopOpacity={0.22}/>
                <stop offset="95%" stopColor="var(--signal)" stopOpacity={0.0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#E4E0D8" />
            <XAxis 
              dataKey="index" 
              stroke="var(--t3)" 
              fontSize={9} 
              tickLine={false}
              tickFormatter={(val) => `S${val + 1}`}
            />
            <YAxis 
              stroke="var(--t3)" 
              fontSize={9} 
              tickLine={false} 
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey="score" 
              name="Perplexity Signal"
              stroke="var(--signal)" 
              strokeWidth={2}
              fill="url(#colorSignal)"
              activeDot={{ r: 5, fill: "var(--signal)", stroke: "var(--surface)", strokeWidth: 1.5 }}
              dot={false}
            />
            {boundaryLines.map((boundaryIdx) => (
              <ReferenceLine 
                key={`ref-line-${boundaryIdx}`}
                x={boundaryIdx} 
                stroke="var(--boundary)" 
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={{ 
                  value: `⚑ S${boundaryIdx + 1}`, 
                  position: 'top', 
                  fill: 'var(--boundary)', 
                  fontSize: 8,
                  fontWeight: 'bold',
                  offset: 5
                }}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
