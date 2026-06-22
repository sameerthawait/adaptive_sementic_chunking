import React from 'react';

export function Badge({ children, variant = 'neutral', className = '' }) {
  const baseStyle = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border transition-colors";
  
  const variantStyles = {
    neutral: "bg-surface-2 text-t2 border-border",
    info: "bg-signal-light text-signal border-signal-border",
    warning: "bg-boundary-light text-boundary border-boundary/30",
    success: "bg-success/10 text-success border-success/30",
    danger: "bg-boundary/10 text-boundary border-boundary/30"
  };

  return (
    <span className={`${baseStyle} ${variantStyles[variant]} ${className}`}>
      {children}
    </span>
  );
}
