import React from 'react';

export function Spinner({ size = 'md', className = '' }) {
  const sizeStyles = {
    sm: "w-4 h-4 border-2",
    md: "w-8 h-8 border-[3px]",
    lg: "w-12 h-12 border-4",
  };

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div 
        className={`animate-spin rounded-full border-t-signal border-r-transparent border-b-transparent border-l-transparent ${sizeStyles[size] || sizeStyles.md}`}
        style={{ borderTopColor: 'var(--signal)' }}
      />
    </div>
  );
}
