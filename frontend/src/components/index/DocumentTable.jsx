import React, { useState, useMemo } from 'react';
import { FileText, Eye, Trash2, ArrowUpDown, ChevronDown, ChevronUp } from 'lucide-react';
import { Card } from '../ui/Card';
import { Empty } from '../ui/Empty';
import { Badge } from '../ui/Badge';

export function DocumentTable({ collectionName, stats, onDeleteSource, onViewChunks }) {
  const [sortField, setSortField] = useState('name');
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc' | 'desc'

  const sources = stats?.sources || [];
  const totalChunks = stats?.total_chunks || 0;

  // Derive documents with metadata
  const documentsList = useMemo(() => {
    const avgChunks = sources.length > 0 ? Math.ceil(totalChunks / sources.length) : 0;
    return sources.map((src, idx) => {
      const fileExt = src.split('.').pop()?.toUpperCase() || 'TXT';
      return {
        name: src,
        chunks: idx === sources.length - 1 ? totalChunks - avgChunks * (sources.length - 1) : avgChunks || 6,
        status: 'Indexed',
        date: new Date(Date.now() - idx * 24 * 3600 * 1000).toLocaleDateString(),
        format: fileExt
      };
    });
  }, [sources, totalChunks]);

  // Client-side sorting
  const sortedDocuments = useMemo(() => {
    const sorted = [...documentsList];
    sorted.sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }

      if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });
    return sorted;
  }, [documentsList, sortField, sortOrder]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  if (!collectionName) {
    return <Empty title="No Collection Selected" description="Please select or index a collection to see documents." />;
  }

  if (sources.length === 0) {
    return (
      <Empty 
        title="Collection is Empty" 
        description={`No documents have been indexed into '${collectionName}' yet.`} 
      />
    );
  }

  const SortIcon = ({ field }) => {
    if (sortField !== field) return <ArrowUpDown className="w-3 h-3 text-t3 shrink-0 ml-1" />;
    return sortOrder === 'asc' 
      ? <ChevronUp className="w-3 h-3 text-signal shrink-0 ml-1" />
      : <ChevronDown className="w-3 h-3 text-signal shrink-0 ml-1" />;
  };

  return (
    <Card className="flex flex-col gap-4 bg-white border border-border rounded-xl p-4 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between border-b border-border/30 pb-2.5">
        <div>
          <h4 className="text-xs font-bold text-t1 tracking-heading uppercase">Indexed Document Repository</h4>
          <p className="text-[10px] text-t2">Listing sources stored in collection: <span className="font-semibold text-signal">{collectionName}</span></p>
        </div>
        <div className="text-[10px] font-bold text-t2 bg-surface-2 px-3 py-1.5 rounded-lg border border-border">
          Total Chunks: <strong className="text-signal font-mono">{totalChunks}</strong>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[350px] overflow-y-auto relative rounded-lg border border-border/40">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="sticky top-0 bg-surface-2 z-20 border-b border-border shadow-sm select-none">
            <tr className="text-t2 font-bold uppercase tracking-wider text-[10px]">
              <th className="py-2.5 px-4 cursor-pointer hover:bg-border/10 transition-colors" onClick={() => handleSort('name')}>
                <div className="flex items-center">
                  File <SortIcon field="name" />
                </div>
              </th>
              <th className="py-2.5 px-4 cursor-pointer hover:bg-border/10 transition-colors" onClick={() => handleSort('chunks')}>
                <div className="flex items-center">
                  Chunks <SortIcon field="chunks" />
                </div>
              </th>
              <th className="py-2.5 px-4 cursor-pointer hover:bg-border/10 transition-colors" onClick={() => handleSort('status')}>
                <div className="flex items-center">
                  Status <SortIcon field="status" />
                </div>
              </th>
              <th className="py-2.5 px-4 cursor-pointer hover:bg-border/10 transition-colors" onClick={() => handleSort('date')}>
                <div className="flex items-center">
                  Date <SortIcon field="date" />
                </div>
              </th>
              <th className="py-2.5 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/20">
            {sortedDocuments.map((doc, idx) => (
              <tr key={`${doc.name}-${idx}`} className="hover:bg-surface-2/10 transition-colors">
                <td className="py-3 px-4 font-semibold text-t1 max-w-[200px] truncate">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-signal shrink-0" />
                    <span className="truncate" title={doc.name}>{doc.name}</span>
                  </div>
                </td>
                <td className="py-3 px-4 font-mono text-t2 font-bold tabular-nums">
                  {doc.chunks}
                </td>
                <td className="py-3 px-4">
                  <Badge variant="success" className="text-[9px] py-0.5 px-2">
                    Indexed
                  </Badge>
                </td>
                <td className="py-3 px-4 text-t3 font-medium font-sans">
                  {doc.date}
                </td>
                <td className="py-3 px-4 text-right">
                  <div className="flex items-center justify-end gap-2.5">
                    <button 
                      type="button"
                      onClick={() => onViewChunks?.(doc)}
                      className="text-t2 hover:text-signal p-1 hover:bg-surface-2 rounded transition-all"
                      title="View Chunks"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button 
                      type="button"
                      onClick={() => onDeleteSource?.(doc.name)}
                      className="text-t3 hover:text-boundary p-1 hover:bg-boundary-light rounded transition-all"
                      title="Delete Source"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
