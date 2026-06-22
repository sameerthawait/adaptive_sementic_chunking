import React, { useRef, useState } from 'react';
import { Upload, FileText, Check, AlertCircle } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

export function DropZone({ onFilesLoaded, onTextPaste, isLoading }) {
  const fileInputRef = useRef(null);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'paste'
  const [isDragOver, setIsDragOver] = useState(false);
  const [loadedFiles, setLoadedFiles] = useState([]);
  const [error, setError] = useState('');
  
  // Paste states
  const [pastedText, setPastedText] = useState('');
  const [pastedFileName, setPastedFileName] = useState('pasted_document.txt');

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const processFiles = (files) => {
    setError('');
    setIsDragOver(false);
    const txtFiles = Array.from(files).filter(
      (file) => file.name.endsWith('.txt') || file.name.endsWith('.md')
    );

    if (txtFiles.length === 0) {
      setError('Please select valid .txt or .md files.');
      return;
    }

    const promises = txtFiles.map((file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          resolve({
            name: file.name,
            content: e.target.result,
            size: file.size,
          });
        };
        reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
        reader.readAsText(file);
      });
    });

    Promise.all(promises)
      .then((results) => {
        setLoadedFiles(results);
        onFilesLoaded(results);
      })
      .catch((err) => {
        setError(err.message);
      });
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isLoading) return;
    if (e.dataTransfer.files) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files) {
      processFiles(e.target.files);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handlePasteSubmit = () => {
    if (!pastedText.trim()) return;
    onTextPaste(pastedText, pastedFileName);
    setPastedText('');
  };

  return (
    <Card className="flex flex-col gap-4">
      {/* Tab Switcher */}
      <div className="flex border-b border-border/40 pb-1">
        <button
          type="button"
          onClick={() => setActiveTab('upload')}
          className={`px-4 py-2 text-xs font-bold uppercase tracking-wider border-b-2 transition-all ${
            activeTab === 'upload'
              ? 'border-signal text-signal'
              : 'border-transparent text-t3 hover:text-t2'
          }`}
        >
          File Upload
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('paste')}
          className={`px-4 py-2 text-xs font-bold uppercase tracking-wider border-b-2 transition-all ${
            activeTab === 'paste'
              ? 'border-signal text-signal'
              : 'border-transparent text-t3 hover:text-t2'
          }`}
        >
          or paste text directly
        </button>
      </div>

      {activeTab === 'upload' ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={triggerFileInput}
          className={`h-[200px] border-2 border-dashed rounded-xl flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
            isDragOver 
              ? 'border-signal bg-signal-light' 
              : 'border-border bg-surface-2/30 hover:border-signal/50 hover:bg-surface-2/50'
          } ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            multiple
            accept=".txt,.md"
            className="hidden"
          />

          <Upload className={`w-8 h-8 mb-2 transition-colors ${isDragOver ? 'text-signal' : 'text-t3'}`} />
          <h4 className="text-xs font-bold text-t1 mb-0.5 uppercase tracking-wider">Drop .txt or .md files here</h4>
          <p className="text-[10px] text-t2 max-w-xs mb-3">or click to browse local files</p>
          <Button variant="outline" className="text-xs py-1.5 px-3" disabled={isLoading}>
            Browse Files
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-[9px] font-bold text-t3 uppercase tracking-wider">Filename Identifier</label>
            <input
              type="text"
              value={pastedFileName}
              onChange={(e) => setPastedFileName(e.target.value)}
              className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs focus:outline-none"
              placeholder="pasted_document.txt"
              disabled={isLoading}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[9px] font-bold text-t3 uppercase tracking-wider">Raw Text Content</label>
            <textarea
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder="Paste raw text drafts directly to index..."
              className="w-full min-h-[120px] p-3 text-xs font-sans border border-border rounded-lg bg-surface focus:outline-none focus:ring-1 focus:ring-signal focus:border-signal"
              disabled={isLoading}
            />
          </div>
          <Button 
            onClick={handlePasteSubmit}
            disabled={isLoading || !pastedText.trim()}
            variant="primary"
            className="w-full text-xs py-2 font-bold uppercase tracking-wider"
          >
            Index
          </Button>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 text-xs text-boundary bg-boundary-light p-3 rounded-lg border border-boundary/20">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {activeTab === 'upload' && loadedFiles.length > 0 && (
        <div className="border border-border/40 rounded-lg p-3 bg-surface-2/20 flex flex-col gap-2">
          <div className="text-[10px] font-bold text-t2 uppercase tracking-wider mb-1">
            Queued Files ({loadedFiles.length})
          </div>
          <div className="max-h-[150px] overflow-y-auto space-y-1.5 pr-2">
            {loadedFiles.map((file, idx) => (
              <div 
                key={`${file.name}-${idx}`}
                className="flex items-center justify-between text-xs p-2 bg-surface border border-border/40 rounded-lg shadow-sm"
              >
                <div className="flex items-center gap-2 truncate">
                  <FileText className="w-4 h-4 text-signal shrink-0" />
                  <span className="font-semibold text-t1 truncate">{file.name}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[10px] text-t3">{(file.size / 1024).toFixed(1)} KB</span>
                  <Check className="w-4 h-4 text-success" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
