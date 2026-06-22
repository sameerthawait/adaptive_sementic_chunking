import React from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';

export function MetricsTable({ metrics }) {
  const defaultMetrics = [
    { method: 'Adaptive Semantic Chunking (ASC)', p1: '0.92', p3: '0.95', p5: '0.97', mrr: '0.94', coherence: '0.88', boundaryQ: '0.91', isAsc: true },
    { method: 'Fixed-Window (250 chars)', p1: '0.58', p3: '0.65', p5: '0.70', mrr: '0.64', coherence: '0.45', boundaryQ: '0.22', isAsc: false },
    { method: 'Fixed-Window (500 chars)', p1: '0.71', p3: '0.78', p5: '0.82', mrr: '0.76', coherence: '0.58', boundaryQ: '0.35', isAsc: false },
    { method: 'Fixed-Window (1000 chars)', p1: '0.79', p3: '0.84', p5: '0.88', mrr: '0.83', coherence: '0.68', boundaryQ: '0.48', isAsc: false },
  ];

  const data = metrics || defaultMetrics;

  return (
    <Card className="flex flex-col gap-4 bg-white border border-border rounded-xl p-4 shadow-sm overflow-hidden">
      <div className="border-b border-border/30 pb-2">
        <h4 className="text-xs font-bold text-t1 tracking-heading uppercase">Comparative Performance Matrix</h4>
        <p className="text-[10px] text-t2">
          Evaluating ASC boundary decisions against character limits.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-border text-t2 font-bold uppercase tracking-wider bg-surface-2/40">
              <th className="py-2.5 px-4 text-[10px]">Method</th>
              <th className="py-2.5 px-4 text-[10px]">P@1</th>
              <th className="py-2.5 px-4 text-[10px]">P@3</th>
              <th className="py-2.5 px-4 text-[10px]">P@5</th>
              <th className="py-2.5 px-4 text-[10px]">MRR</th>
              <th className="py-2.5 px-4 text-[10px]">Coherence</th>
              <th className="py-2.5 px-4 text-[10px]">Boundary Q.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/20">
            {data.map((row) => (
              <tr 
                key={row.method} 
                className={`hover:bg-surface-2/10 transition-colors ${
                  row.isAsc ? 'bg-signal-light border-l-2 border-l-signal' : ''
                }`}
              >
                <td className="py-3 px-4 font-semibold text-t1 flex items-center gap-1.5">
                  <span>{row.method}</span>
                  {row.isAsc && <Badge variant="success" className="text-[9px] py-0 px-1.5">Best</Badge>}
                </td>
                <td className={`py-3 px-4 font-mono tabular-nums ${row.isAsc ? 'font-bold text-signal' : 'text-t2'}`}>
                  {row.p1}
                </td>
                <td className={`py-3 px-4 font-mono tabular-nums ${row.isAsc ? 'font-bold text-signal' : 'text-t2'}`}>
                  {row.p3}
                </td>
                <td className={`py-3 px-4 font-mono tabular-nums ${row.isAsc ? 'font-bold text-signal' : 'text-t2'}`}>
                  {row.p5}
                </td>
                <td className={`py-3 px-4 font-mono tabular-nums ${row.isAsc ? 'font-bold text-signal' : 'text-t2'}`}>
                  {row.mrr}
                </td>
                <td className={`py-3 px-4 font-mono tabular-nums ${row.isAsc ? 'font-bold text-signal' : 'text-t2'}`}>
                  {row.coherence}
                </td>
                <td className={`py-3 px-4 font-mono tabular-nums ${row.isAsc ? 'font-bold text-signal' : 'text-t2'}`}>
                  {row.boundaryQ}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
