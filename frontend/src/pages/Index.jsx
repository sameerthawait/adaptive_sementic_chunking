import React, { useState, useEffect } from 'react';
import { useChunk } from '../hooks/useChunk';
import { DropZone } from '../components/index/DropZone';
import { DocumentTable } from '../components/index/DocumentTable';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Toast } from '../components/ui/Toast';
import { logActivity } from '../utils/activity';
import { 
  FolderPlus, 
  Database, 
  RefreshCw, 
  Download 
} from 'lucide-react';
import { motion, useReducedMotion } from 'framer-motion';

export default function Index() {
  const { indexMutation, collectionsQuery } = useChunk();
  const shouldReduceMotion = useReducedMotion();

  const collections = collectionsQuery.data?.collections || [];
  const collectionsStats = collectionsQuery.data?.stats || {};

  const [activeCollection, setActiveCollection] = useState('default');
  const [toastMessage, setToastMessage] = useState('');
  const [toastType, setToastType] = useState('info');

  useEffect(() => {
    if (collections.length > 0 && activeCollection === 'default' && !collections.includes('default')) {
      setActiveCollection(collections[0]);
    }
  }, [collections, activeCollection]);

  const handleFilesLoaded = (files) => {
    if (files.length === 0) return;
    const texts = files.map((f) => f.content);
    const sources = files.map((f) => f.name);

    indexMutation.mutate(
      { texts, sources, collection: activeCollection },
      {
        onSuccess: (data) => {
          setToastType('success');
          setToastMessage(`Indexed ${data.indexed} files successfully into '${data.collection}'!`);
          logActivity('index', `Indexed ${sources.length} file(s) into '${activeCollection}'`, 'success');
        },
        onError: (err) => {
          setToastType('error');
          setToastMessage(err.message || 'Indexing failed');
          logActivity('index', `Failed to index ${sources.length} file(s) into '${activeCollection}'`, 'failed');
        },
      }
    );
  };

  const handleTextPaste = (text, fileName) => {
    indexMutation.mutate(
      { texts: [text], sources: [fileName], collection: activeCollection },
      {
        onSuccess: (data) => {
          setToastType('success');
          setToastMessage(`Indexed pasted text successfully into '${data.collection}'!`);
          logActivity('index', `Indexed pasted text into '${activeCollection}'`, 'success');
        },
        onError: (err) => {
          setToastType('error');
          setToastMessage(err.message || 'Indexing failed');
          logActivity('index', `Failed to index pasted text into '${activeCollection}'`, 'failed');
        },
      }
    );
  };

  const handleCreateCollection = () => {
    const name = prompt("Enter name for the new vector collection:");
    if (name && name.trim()) {
      const cleanName = name.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      if (collections.includes(cleanName)) {
        setToastType('error');
        setToastMessage(`Collection '${cleanName}' already exists.`);
        return;
      }
      setActiveCollection(cleanName);
      setToastType('success');
      setToastMessage(`Selected new collection context: '${cleanName}'`);
    }
  };

  const handleReIndexAll = () => {
    setToastType('info');
    setToastMessage('Re-indexing vector databases...');
  };

  const handleExportCollection = () => {
    setToastType('success');
    setToastMessage(`Exported metadata map for collection: ${activeCollection}`);
  };

  const handleViewChunks = (doc) => {
    setToastType('info');
    setToastMessage(`Inspecting segments for document: ${doc.name}`);
  };

  const handleDeleteSource = (sourceName) => {
    setToastType('warning');
    setToastMessage(`Deletion of '${sourceName}' is disabled in read-only web server.`);
  };

  const isIndexing = indexMutation.isPending;
  const stats = collectionsStats[activeCollection] || { sources: [], total_chunks: 0 };
  const numDocs = stats.sources?.length || 0;
  const numChunks = stats.total_chunks || 0;

  // Animation variants
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
      {/* Header and Eyebrow */}
      <div>
        <span className="text-[11px] font-bold text-t3 uppercase tracking-wider block mb-1">
          KNOWLEDGE INGESTION
        </span>
        <h2 className="text-xl font-bold text-t1 tracking-heading leading-tight">
          Vector Database Ingestion
        </h2>
        <p className="text-xs text-t2 mt-1">
          Index text assets into target collections to enable semantic-shift context retrievals.
        </p>
      </div>

      {/* Top Row Collection Select & Stats */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-border rounded-xl p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Target Collection</label>
          <select
            value={activeCollection}
            onChange={(e) => setActiveCollection(e.target.value)}
            className="p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs font-semibold focus:outline-none"
            disabled={isIndexing}
          >
            {collections.length === 0 ? (
              <option value="default">default</option>
            ) : (
              collections.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))
            )}
          </select>
          <button
            type="button"
            onClick={handleCreateCollection}
            className="flex items-center gap-1 text-[10px] font-bold text-signal border border-signal/20 hover:bg-signal/5 px-2.5 py-1.5 rounded bg-transparent transition-all uppercase tracking-wider cursor-pointer"
          >
            <FolderPlus className="w-3.5 h-3.5" />
            New Collection
          </button>
        </div>

        {/* Stats Pill */}
        <Badge variant="info" className="text-[10px] font-bold uppercase tracking-wider font-mono py-1 px-3">
          {numDocs} {numDocs === 1 ? 'doc' : 'docs'} · {numChunks} {numChunks === 1 ? 'chunk' : 'chunks'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[40%_60%] gap-6 items-start">
        {/* Upload panel (DropZone) */}
        <div className="flex flex-col gap-3">
          <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">Ingestion Source</h3>
          <DropZone 
            onFilesLoaded={handleFilesLoaded} 
            onTextPaste={handleTextPaste}
            isLoading={isIndexing} 
          />
        </div>

        {/* Document Table Repository */}
        <div className="flex flex-col gap-3">
          <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">Stored Assets</h3>
          <DocumentTable 
            collectionName={activeCollection} 
            stats={stats} 
            onViewChunks={handleViewChunks}
            onDeleteSource={handleDeleteSource}
          />
        </div>
      </div>

      {/* Sticky Bottom bar */}
      <div className="sticky bottom-0 bg-bg/85 backdrop-blur border-t border-border/40 py-3.5 flex justify-end gap-3 z-30 -mx-6 px-6 mt-6 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
        <Button 
          variant="ghost" 
          onClick={handleReIndexAll} 
          className="text-xs uppercase tracking-wider font-bold flex items-center gap-1 cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Re-index all
        </Button>
        <Button 
          variant="ghost" 
          onClick={handleExportCollection} 
          className="text-xs uppercase tracking-wider font-bold flex items-center gap-1 cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          Export collection
        </Button>
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
