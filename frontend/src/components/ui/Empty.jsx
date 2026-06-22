import React from 'react';
import { Inbox } from 'lucide-react';

export function Empty({ 
  title = "No data found", 
  description = "Get started by executing a task or uploading a document.", 
  icon = <Inbox className="w-8 h-8 text-t3" />, 
  className = "" 
}) {
  return (
    <div className={`flex flex-col items-center justify-center text-center p-8 border border-dashed border-border rounded-xl bg-surface-2/40 ${className}`}>
      <div className="p-3 bg-surface border border-border rounded-full shadow-sm mb-3">
        {icon}
      </div>
      <h4 className="text-sm font-semibold text-t1 mb-1">{title}</h4>
      <p className="text-xs text-t2 max-w-xs">{description}</p>
    </div>
  );
}
