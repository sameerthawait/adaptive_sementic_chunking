import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { useHealth } from '../../hooks/useHealth';

export function Layout() {
  const { data: healthStatus } = useHealth();

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <Sidebar healthStatus={healthStatus} />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar healthStatus={healthStatus} />

        <main className="flex-1 overflow-y-auto p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
