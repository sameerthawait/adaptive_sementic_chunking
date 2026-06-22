import React, { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useHealth } from '../hooks/useHealth';
import { useChunk } from '../hooks/useChunk';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Empty } from '../components/ui/Empty';
import { Spinner } from '../components/ui/Spinner';
import { getActivities } from '../utils/activity';
import { 
  Scissors, 
  MessageSquare, 
  Database, 
  HelpCircle, 
  TrendingUp, 
  FileText, 
  MessageCircle,
  FolderSync,
  AlertCircle,
  Activity
} from 'lucide-react';
import { motion, useReducedMotion } from 'framer-motion';

export default function Dashboard() {
  const navigate = useNavigate();
  const { data: health, isLoading: healthLoading } = useHealth();
  const { collectionsQuery } = useChunk();
  const { data: collectionsData, isLoading: collectionsLoading } = collectionsQuery;

  const [activities, setActivities] = useState([]);
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    setActivities(getActivities().slice(0, 5));
  }, []);

  const collections = collectionsData?.collections || [];
  const stats = collectionsData?.stats || {};

  // Calculate totals
  let totalChunks = 0;
  collections.forEach((col) => {
    totalChunks += stats[col]?.total_chunks || 0;
  });

  const isConnected = health?.status === 'ok';
  const lastRunTime = activities[0]?.timestamp || 'N/A';

  if (healthLoading || collectionsLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  const activityIcons = {
    chunk: <Scissors className="w-4 h-4 text-signal shrink-0" />,
    index: <Database className="w-4 h-4 text-t2 shrink-0" />,
    query: <MessageCircle className="w-4 h-4 text-boundary shrink-0" />,
  };



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
      className="flex flex-col gap-6 select-none"
    >
      {/* Header and eyebrow */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <span className="text-[11px] font-bold text-t3 uppercase tracking-wider block mb-1">
            SEMANTIC CHUNKER
          </span>
          <h2 className="text-xl font-bold text-t1 tracking-heading leading-tight">
            Adaptive Semantic Chunking
          </h2>
        </div>

        {/* API Health Pill */}
        <Badge variant={isConnected ? 'success' : 'danger'} className="text-[10px] font-bold uppercase tracking-wider py-1 px-3">
          {isConnected ? 'API Online' : 'API Offline'}
        </Badge>
      </div>

      {/* Stats row (4 cards) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4 flex flex-col gap-1 bg-surface border-border">
          <span className="text-[11px] font-bold text-t3 uppercase tracking-wider">Total Chunks</span>
          <span className="text-2xl font-bold text-t1 leading-none tracking-tight">{totalChunks}</span>
        </Card>

        <Card className="p-4 flex flex-col gap-1 bg-surface border-border">
          <span className="text-[11px] font-bold text-t3 uppercase tracking-wider">Collections</span>
          <span className="text-2xl font-bold text-t1 leading-none tracking-tight">{collections.length}</span>
        </Card>

        <Card className="p-4 flex flex-col gap-1 bg-surface border-border">
          <span className="text-[11px] font-bold text-t3 uppercase tracking-wider">Avg Coherence</span>
          <span className="text-2xl font-bold text-signal leading-none tracking-tight">0.884</span>
        </Card>

        <Card className="p-4 flex flex-col gap-1 bg-surface border-border">
          <span className="text-[11px] font-bold text-t3 uppercase tracking-wider">Last Run</span>
          <span className="text-2xl font-bold text-t1 leading-none tracking-tight font-sans text-ellipsis overflow-hidden whitespace-nowrap">{lastRunTime}</span>
        </Card>
      </div>

      {/* Two large clickable navigation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div 
          onClick={() => navigate('/chunker')}
          className="bg-surface border border-border rounded-xl p-6 cursor-pointer hover:border-signal hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 flex flex-col gap-2 group"
        >
          <div className="w-10 h-10 rounded-lg bg-signal-light border border-signal-border flex items-center justify-center text-signal group-hover:scale-105 transition-transform">
            <Scissors className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-sm text-t1 tracking-heading mt-2 group-hover:text-signal transition-colors">
            Chunk a Document
          </h3>
          <p className="text-xs text-t2 leading-relaxed">
            Upload text drafts to test perplexity shifts, preview sentence segmentations, and export boundaries metrics.
          </p>
        </div>

        <div 
          onClick={() => navigate('/query')}
          className="bg-surface border border-border rounded-xl p-6 cursor-pointer hover:border-signal hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 flex flex-col gap-2 group"
        >
          <div className="w-10 h-10 rounded-lg bg-boundary-light border border-boundary/20 flex items-center justify-center text-boundary group-hover:scale-105 transition-transform">
            <MessageSquare className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-sm text-t1 tracking-heading mt-2 group-hover:text-boundary transition-colors">
            Ask a Question
          </h3>
          <p className="text-xs text-t2 leading-relaxed">
            Query collections using the agentic self-correcting RAG pipeline, analyze context grading, and review reference lists.
          </p>
        </div>
      </div>

      {/* Recent activity list */}
      <div className="flex flex-col gap-3">
        <h3 className="text-xs font-bold text-t2 uppercase tracking-wider">Recent Activity</h3>
        
        {activities.length === 0 ? (
          <Empty 
            title="No operations yet" 
            description="Start by chunking a document or indexing texts to populate recent logs."
            icon={<Activity className="w-6 h-6 text-t3 animate-pulse" />}
          />
        ) : (
          <div className="bg-surface border border-border rounded-xl divide-y divide-border/20 overflow-hidden shadow-sm">
            {activities.map((act) => (
              <div key={act.id} className="p-4 flex items-center justify-between gap-4 hover:bg-surface-2/20 transition-colors">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="p-2 bg-surface-2 border border-border/30 rounded-lg shrink-0">
                    {activityIcons[act.type] || <FileText className="w-4 h-4 text-t3" />}
                  </div>
                  <div className="min-w-0">
                    <span className="font-semibold text-xs text-t1 truncate block">
                      {act.details}
                    </span>
                    <span className="text-[10px] text-t3 font-medium uppercase mt-0.5 block">
                      {act.type} Operation
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-[10px] text-t2 font-medium font-sans">
                    {act.timestamp}
                  </span>
                  <Badge variant={act.status === 'success' ? 'success' : 'danger'}>
                    {act.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
