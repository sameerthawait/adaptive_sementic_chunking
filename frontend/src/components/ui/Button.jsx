import React from 'react';

export function Button({ 
  children, 
  variant = 'primary', 
  onClick, 
  disabled = false, 
  type = 'button',
  className = '', 
  ...props 
}) {
  const baseStyle = "inline-flex items-center justify-center font-medium transition-all duration-200 border rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none";
  
  const sizeStyle = "px-4 py-2 text-sm leading-5";
  
  const variantStyles = {
    primary: "bg-signal text-white border-transparent hover:bg-signal/90 focus:ring-signal",
    secondary: "bg-surface-2 text-t1 border-border hover:bg-border/20 focus:ring-border",
    outline: "bg-transparent text-t2 border-border hover:bg-surface-2 hover:text-t1 focus:ring-border",
    danger: "bg-boundary text-white border-transparent hover:bg-boundary/90 focus:ring-boundary",
    success: "bg-success text-white border-transparent hover:bg-opacity-95 focus:ring-success",
    ghost: "bg-transparent text-t2 border-transparent hover:bg-surface-2 hover:text-t1 focus:ring-border"
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyle} ${sizeStyle} ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
