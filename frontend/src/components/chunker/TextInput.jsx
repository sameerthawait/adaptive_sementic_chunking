import React from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

export function TextInput({ 
  text, 
  setText, 
  source, 
  setSource, 
  visualize, 
  setVisualize, 
  onSubmit, 
  isLoading 
}) {
  return (
    <Card className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-bold text-t2 uppercase tracking-wider">Document Text Source</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or write your structured text drafts here (e.g., research papers, documentation, articles)..."
          className="w-full min-h-[160px] p-3 text-xs border border-border rounded-lg bg-surface-2 focus:bg-white focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal resize-y"
          disabled={isLoading}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-bold text-t2 uppercase tracking-wider">Document Identifier</label>
          <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal"
            placeholder="document_source.txt"
            disabled={isLoading}
          />
        </div>

        <div className="flex items-center gap-2 select-none pt-4">
          <input
            type="checkbox"
            id="visualize_checkbox"
            checked={visualize}
            onChange={(e) => setVisualize(e.target.checked)}
            className="w-4 h-4 rounded text-signal border-border focus:ring-signal"
            disabled={isLoading}
          />
          <label htmlFor="visualize_checkbox" className="text-xs font-bold text-t2 uppercase tracking-wider cursor-pointer">
            Calculate Diagnostic Statistics
          </label>
        </div>
      </div>

      <div className="pt-2">
        <Button 
          onClick={onSubmit} 
          disabled={isLoading || !text.trim()} 
          variant="primary"
          className="w-full md:w-auto text-xs"
        >
          {isLoading ? 'Running Segmentation...' : '⚡ Segment Document'}
        </Button>
      </div>
    </Card>
  );
}
