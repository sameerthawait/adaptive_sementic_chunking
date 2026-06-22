import React from 'react';
import { Cpu, Server } from 'lucide-react';
import { Badge } from '../ui/Badge';

export function TopBar({ healthStatus }) {
  const provider = healthStatus?.provider || 'unknown';
  const models = healthStatus?.models_loaded || [];
  
  return (
    <header className="h-16 border-b border-border bg-surface flex items-center justify-between px-8 shrink-0 select-none">
      <div className="flex items-center gap-4">
        <h1 className="text-base font-bold text-t1 tracking-heading">Adaptive Semantic Chunking</h1>
      </div>
      
      <div className="flex items-center gap-6 text-xs">
        {/* ChromaDB Status */}
        <div className="flex items-center gap-2 text-t2 font-medium">
          <Server className="w-3.5 h-3.5 text-t3" />
          <span>Vector DB:</span>
          <Badge variant={healthStatus?.chromadb ? 'success' : 'danger'}>
            {healthStatus?.chromadb ? 'Chroma Active' : 'Offline'}
          </Badge>
        </div>

        {/* Model Provider Info */}
        <div className="flex items-center gap-2 text-t2 font-medium">
          <Cpu className="w-3.5 h-3.5 text-t3" />
          <span>Provider:</span>
          <Badge variant={healthStatus?.provider_connected ? 'info' : 'neutral'} className="uppercase">
            {provider}
          </Badge>
        </div>

        {/* Loaded Models List */}
        {models.length > 0 && (
          <div className="hidden md:flex items-center gap-2 text-t2">
            <span className="text-t3">Model:</span>
            <span className="font-mono text-[10px] bg-surface-2 px-2 py-0.5 rounded border border-border">
              {models[0]}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
