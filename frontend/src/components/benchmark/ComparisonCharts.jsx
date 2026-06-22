import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';
import { Card } from '../ui/Card';

export function ComparisonCharts({ chartData }) {
  // Datasets for 2x2 Grid Charts
  const precisionData = [
    { name: 'ASC', value: 0.94, isAsc: true },
    { name: 'F-256', value: 0.64, isAsc: false },
    { name: 'F-512', value: 0.76, isAsc: false },
    { name: 'F-1024', value: 0.83, isAsc: false },
  ];

  const coherenceData = [
    { name: 'ASC', value: 0.88, isAsc: true },
    { name: 'F-256', value: 0.45, isAsc: false },
    { name: 'F-512', value: 0.58, isAsc: false },
    { name: 'F-1024', value: 0.68, isAsc: false },
  ];

  const boundaryData = [
    { name: 'ASC', value: 0.91, isAsc: true },
    { name: 'F-256', value: 0.22, isAsc: false },
    { name: 'F-512', value: 0.35, isAsc: false },
    { name: 'F-1024', value: 0.48, isAsc: false },
  ];

  const sizeVarianceData = [
    { name: 'ASC', value: 280, isAsc: true },
    { name: 'F-256', value: 5, isAsc: false },
    { name: 'F-512', value: 8, isAsc: false },
    { name: 'F-1024', value: 12, isAsc: false },
  ];

  const renderBarChart = (title, data, unit = "") => {
    return (
      <Card className="flex flex-col gap-2 p-3 bg-white border border-border rounded-lg shadow-sm">
        <h5 className="text-[10px] font-bold text-t2 uppercase tracking-wider">{title}</h5>
        <div className="h-[140px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 5, right: 5, left: -30, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E4E0D8" opacity={0.5} />
              <XAxis dataKey="name" stroke="var(--t3)" fontSize={9} tickLine={false} />
              <YAxis stroke="var(--t3)" fontSize={9} tickLine={false} />
              <Tooltip 
                contentStyle={{ fontSize: '10px', padding: '4px 8px', borderRadius: '6px' }}
                formatter={(val) => [`${val}${unit}`, 'Value']}
              />
              <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                {data.map((entry, idx) => (
                  <Cell 
                    key={`cell-${idx}`} 
                    fill={entry.isAsc ? 'var(--signal)' : '#A09890'} 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    );
  };

  return (
    <div className="flex flex-col gap-3 w-full">
      <div className="border-b border-border/30 pb-1.5">
        <h4 className="text-xs font-bold text-t2 uppercase tracking-wider">Evaluation Metrics Charts</h4>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
        {renderBarChart("Precision @ K (Retrieval Match)", precisionData, "%")}
        {renderBarChart("Semantic Coherence Score", coherenceData)}
        {renderBarChart("Boundary Quality Index", boundaryData)}
        {renderBarChart("Context Length StdDev (Adaptive Size)", sizeVarianceData, " ch")}
      </div>
    </div>
  );
}
