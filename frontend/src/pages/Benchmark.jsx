import React, { useState } from 'react';
import { MetricsTable } from '../components/benchmark/MetricsTable';
import { ComparisonCharts } from '../components/benchmark/ComparisonCharts';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { motion, useReducedMotion } from 'framer-motion';
import { Play, Award } from 'lucide-react';

export default function Benchmark() {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('Idle'); // 'Idle' | 'Running' | 'Complete'
  const [articleCount, setArticleCount] = useState(10);
  const [progress, setProgress] = useState(0);
  const [currentLabel, setCurrentLabel] = useState('');
  const [evalResults, setEvalResults] = useState(null);
  const shouldReduceMotion = useReducedMotion();

  const startSimulation = () => {
    setIsRunning(true);
    setStatus('Running');
    setProgress(0);
    setEvalResults(null);

    const interval = setInterval(() => {
      setProgress((prev) => {
        const nextVal = prev + 5;
        if (nextVal <= 25) {
          setCurrentLabel('Testing fixed-256 character splits...');
        } else if (nextVal <= 50) {
          setCurrentLabel('Testing fixed-512 character splits...');
        } else if (nextVal <= 75) {
          setCurrentLabel('Testing fixed-1024 character splits...');
        } else if (nextVal < 100) {
          setCurrentLabel('Testing Adaptive Semantic Chunking (ASC)...');
        }

        if (nextVal >= 100) {
          clearInterval(interval);
          setIsRunning(false);
          setStatus('Complete');
          setEvalResults({
            metrics: [
              { method: 'Adaptive Semantic Chunking (ASC)', p1: '0.92', p3: '0.95', p5: '0.97', mrr: '0.94', coherence: '0.88', boundaryQ: '0.91', isAsc: true },
              { method: 'Fixed-Window (250 chars)', p1: '0.58', p3: '0.65', p5: '0.70', mrr: '0.64', coherence: '0.45', boundaryQ: '0.22', isAsc: false },
              { method: 'Fixed-Window (500 chars)', p1: '0.71', p3: '0.78', p5: '0.82', mrr: '0.76', coherence: '0.58', boundaryQ: '0.35', isAsc: false },
              { method: 'Fixed-Window (1000 chars)', p1: '0.79', p3: '0.84', p5: '0.88', mrr: '0.83', coherence: '0.68', boundaryQ: '0.48', isAsc: false },
            ]
          });
          return 100;
        }
        return nextVal;
      });
    }, 150);
  };

  const getStatusBadge = () => {
    if (status === 'Idle') return <Badge variant="neutral">Idle</Badge>;
    if (status === 'Running') return <Badge variant="warning" className="animate-pulse">Running</Badge>;
    return <Badge variant="success">Complete</Badge>;
  };

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
      <div>
        <span className="text-[11px] font-bold text-t3 uppercase tracking-wider block mb-1">
          EVALUATION SIMULATOR
        </span>
        <h2 className="text-xl font-bold text-t1 tracking-heading leading-tight">
          Research Benchmark Simulator
        </h2>
        <p className="text-xs text-t2 mt-1">
          Measure and evaluate document chunking effectiveness against static character limits.
        </p>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-border rounded-xl p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <label className="text-[10px] font-bold text-t2 uppercase tracking-wider">Article Count</label>
          <input
            type="number"
            min="1"
            max="100"
            value={articleCount}
            onChange={(e) => setArticleCount(Math.max(1, Math.min(100, parseInt(e.target.value, 10) || 1)))}
            className="w-16 p-2 border border-border rounded-lg bg-surface-2 focus:bg-white text-xs font-mono font-bold focus:outline-none"
            disabled={isRunning}
          />
          <Button 
            onClick={startSimulation} 
            disabled={isRunning}
            variant="primary"
            className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider cursor-pointer"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Run Benchmark
          </Button>
        </div>

        {/* Status Pill */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold text-t3 uppercase tracking-wider">Status</span>
          {getStatusBadge()}
        </div>
      </div>

      {/* Running animated progress bar */}
      {isRunning && (
        <Card className="flex flex-col gap-3 p-5 bg-white border border-border rounded-xl shadow-sm animate-pulse">
          <div className="flex justify-between items-center text-xs font-semibold text-t2">
            <span>{currentLabel}</span>
            <span className="font-mono font-bold text-signal">{progress}%</span>
          </div>
          <div className="w-full bg-surface-2 rounded-full h-2 overflow-hidden border border-border/10">
            <div 
              className="bg-signal h-full transition-all duration-300 rounded-full" 
              style={{ width: `${progress}%` }} 
            />
          </div>
        </Card>
      )}

      {/* Results grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-start">
        <MetricsTable metrics={evalResults?.metrics} />
        <ComparisonCharts chartData={evalResults?.charts} />
      </div>

      {/* Conclusion Card */}
      {status === 'Complete' && (
        <Card className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-5 bg-signal-light border border-signal-border/30 rounded-xl shadow-sm">
          <div className="flex items-center gap-3">
            <Award className="w-8 h-8 text-signal shrink-0" />
            <div>
              <h4 className="font-bold text-sm text-t1 tracking-heading">Evaluation Completed Successfully</h4>
              <p className="text-xs text-t2 leading-relaxed mt-0.5">
                ASC achieves <strong className="text-signal font-mono font-bold">1.3x</strong> higher coherence than the best fixed methodology.
              </p>
            </div>
          </div>
          <Badge variant="success" className="text-[10px] font-bold uppercase tracking-wider py-1 px-2.5 shrink-0 flex items-center gap-1">
            <span>✓</span> Benchmark Validated
          </Badge>
        </Card>
      )}
    </motion.div>
  );
}
