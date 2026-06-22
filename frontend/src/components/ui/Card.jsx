import React from 'react';

export function Card({ children, className = '', ...props }) {
  return (
    <div 
      className={`bg-surface border border-border rounded-xl shadow-sm p-6 transition-all duration-200 ${className}`} 
      {...props}
    >
      {children}
    </div>
  );
}
