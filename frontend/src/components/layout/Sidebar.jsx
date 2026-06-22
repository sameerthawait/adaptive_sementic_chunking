import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Scissors, MessageSquare, Database, BarChart3 } from 'lucide-react';

export function Sidebar({ healthStatus }) {
  const navItems = [
    { name: 'Dashboard', path: '/', icon: <LayoutDashboard className="w-4 h-4" /> },
    { name: 'Chunker', path: '/chunker', icon: <Scissors className="w-4 h-4" /> },
    { name: 'Query', path: '/query', icon: <MessageSquare className="w-4 h-4" /> },
    { name: 'Index', path: '/index', icon: <Database className="w-4 h-4" /> },
    { name: 'Benchmark', path: '/benchmark', icon: <BarChart3 className="w-4 h-4" /> },
  ];

  const isConnected = healthStatus?.status === 'ok';

  return (
    <aside className="w-[220px] bg-nav text-t3 flex flex-col h-screen shrink-0 border-r border-border/10 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-border/10 flex items-center gap-2.5">
        <div className="bg-signal w-8 h-8 rounded flex items-center justify-center text-white font-bold text-sm shadow-md">
          A
        </div>
        <div className="min-w-0">
          <h2 className="text-white font-bold text-sm leading-none tracking-heading">ASC</h2>
          <span className="text-[10px] text-t3 tracking-wider font-semibold uppercase block mt-0.5">Semantic Chunker</span>
        </div>
      </div>

      {/* Nav List */}
      <nav className="flex-1 px-2.5 py-5 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `
              flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-medium transition-all duration-150
              ${isActive 
                ? 'bg-signal-light text-signal border-l-[3px] border-signal rounded-l-none font-semibold' 
                : 'hover:bg-nav-hover/40 hover:text-white'
              }
            `}
          >
            {item.icon}
            <span>{item.name}</span>
          </NavLink>
        ))}
      </nav>

      {/* Health Status Dot at Bottom */}
      <div className="p-4 border-t border-border/10 bg-nav-hover/10 flex items-center justify-between text-[11px]">
        <span className="text-t3 font-medium">System status:</span>
        <div className="flex items-center gap-1.5 font-medium text-white">
          <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success' : 'bg-boundary'}`} />
          <span className="capitalize">{isConnected ? 'Connected' : 'Offline'}</span>
        </div>
      </div>
    </aside>
  );
}
