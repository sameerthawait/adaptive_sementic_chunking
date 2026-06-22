import React from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Cpu, CheckCircle2 } from 'lucide-react';

export function AnswerCard({ answer, entailmentScore, iterations, useRag }) {
  if (!answer) return null;

  return (
    <Card className="flex flex-col gap-4 border-l-4 border-l-signal">
      <div className="flex items-center justify-between border-b border-border/30 pb-2">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-signal" />
          <h4 className="font-bold text-sm text-t1 tracking-heading">
            {useRag ? 'Agent Answer' : 'Retrieval Result Summary'}
          </h4>
        </div>

        {useRag && (
          <div className="flex items-center gap-1.5">
            {entailmentScore !== undefined && (
              <Badge variant={entailmentScore >= 0.8 ? 'success' : 'warning'}>
                Entailment: {parseFloat(entailmentScore).toFixed(2)}
              </Badge>
            )}
            {iterations !== undefined && (
              <Badge variant="info">
                Loops: {iterations}
              </Badge>
            )}
          </div>
        )}
      </div>

      <div className="text-xs text-t1 leading-relaxed bg-surface-2/40 p-4 rounded-xl border border-border/20 font-serif whitespace-pre-wrap">
        {answer}
      </div>

      {useRag && entailmentScore >= 0.8 && (
        <div className="flex items-center gap-1.5 text-success text-[10px] font-bold uppercase tracking-wider">
          <CheckCircle2 className="w-4 h-4" />
          <span>Strictly Grounded in Sources</span>
        </div>
      )}
    </Card>
  );
}
