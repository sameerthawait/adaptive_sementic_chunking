import React, { useEffect } from 'react';
import { X, CheckCircle, AlertOctagon, Info } from 'lucide-react';
import { motion, useReducedMotion } from 'framer-motion';

export function Toast({ message, type = 'info', onClose, duration = 4000 }) {
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    const timer = setTimeout(() => {
      onClose?.();
    }, duration);
    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const icons = {
    success: <CheckCircle className="w-5 h-5 text-success" />,
    error: <AlertOctagon className="w-5 h-5 text-boundary" />,
    info: <Info className="w-5 h-5 text-signal" />,
  };

  const bgColors = {
    success: "bg-white border-success/30 text-t1",
    error: "bg-boundary-light border-boundary/30 text-boundary",
    info: "bg-signal-light border-signal-border text-t1",
  };

  return (
    <motion.div 
      initial={{ opacity: 0, x: shouldReduceMotion ? 0 : 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: shouldReduceMotion ? 0 : 300 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className={`fixed top-5 right-5 z-50 flex items-center gap-3 p-4 border rounded-xl shadow-lg ${bgColors[type] || bgColors.info}`}
    >
      {icons[type]}
      <span className="text-sm font-medium">{message}</span>
      <button 
        onClick={onClose}
        className="p-1 rounded-lg hover:bg-black/5 transition-colors ml-2"
      >
        <X className="w-4 h-4" />
      </button>
    </motion.div>
  );
}
