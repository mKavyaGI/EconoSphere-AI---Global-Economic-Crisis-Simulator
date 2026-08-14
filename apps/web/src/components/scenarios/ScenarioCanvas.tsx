'use client';

import React, { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useTheme } from 'next-themes';
import { ScenarioEvent } from '@/hooks/useScenarios';

interface ScenarioCanvasProps {
  events: ScenarioEvent[];
  onEventSelect: (event: ScenarioEvent | null) => void;
}

const categoryColors: Record<string, string> = {
  ECONOMIC: '#3b82f6', // blue
  POLITICAL: '#8b5cf6', // purple
  ENVIRONMENTAL: '#10b981', // green
  FINANCIAL: '#eab308', // yellow
  HEALTH: '#ef4444', // red
  ENERGY: '#f97316', // orange
  TECHNOLOGY: '#06b6d4', // cyan
  'SUPPLY_CHAIN': '#64748b', // slate
};

export default function ScenarioCanvas({ events, onEventSelect }: ScenarioCanvasProps) {
  const { theme } = useTheme();

  const initialNodes: Node[] = useMemo(() => {
    // Simple layout strategy: roots at top, children below
    const nodes: Node[] = [];
    const levelMap = new Map<number, number>(); // eventId -> level
    
    // Find roots
    const roots = events.filter(e => !e.parent_event_id);
    roots.forEach((root, idx) => {
      levelMap.set(root.id, 0);
      nodes.push({
        id: root.id.toString(),
        position: { x: 250 * idx, y: 50 },
        data: { label: root.title, event: root },
        style: {
          background: theme === 'dark' ? '#1e293b' : '#ffffff',
          color: theme === 'dark' ? '#f8fafc' : '#0f172a',
          border: `2px solid ${categoryColors[root.category] || '#94a3b8'}`,
          borderRadius: '8px',
          padding: '10px',
          width: 180,
        },
      });
    });

    // Find children recursively (simple tree layout)
    const processChildren = (parentId: number, level: number, parentX: number) => {
      const children = events.filter(e => e.parent_event_id === parentId);
      children.forEach((child, idx) => {
        levelMap.set(child.id, level);
        nodes.push({
          id: child.id.toString(),
          position: { x: parentX + (idx - (children.length - 1) / 2) * 200, y: 50 + level * 150 },
          data: { label: child.title, event: child },
          style: {
            background: theme === 'dark' ? '#1e293b' : '#ffffff',
            color: theme === 'dark' ? '#f8fafc' : '#0f172a',
            border: `2px solid ${categoryColors[child.category] || '#94a3b8'}`,
            borderRadius: '8px',
            padding: '10px',
            width: 180,
          },
        });
        processChildren(child.id, level + 1, parentX + (idx - (children.length - 1) / 2) * 200);
      });
    };

    roots.forEach((root, idx) => {
      processChildren(root.id, 1, 250 * idx);
    });

    return nodes;
  }, [events, theme]);

  const initialEdges: Edge[] = useMemo(() => {
    return events
      .filter(e => e.parent_event_id !== null)
      .map(e => ({
        id: `e${e.parent_event_id}-${e.id}`,
        source: e.parent_event_id!.toString(),
        target: e.id.toString(),
        animated: true,
        style: { stroke: theme === 'dark' ? '#cbd5e1' : '#475569' },
      }));
  }, [events, theme]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  React.useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const onConnect = useCallback(
    (params: Connection | Edge) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    if (node.data && node.data.event) {
      onEventSelect(node.data.event as ScenarioEvent);
    }
  }, [onEventSelect]);

  const onPaneClick = useCallback(() => {
    onEventSelect(null);
  }, [onEventSelect]);

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '500px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        colorMode={theme === 'dark' ? 'dark' : 'light'}
      >
        <Controls />
        <MiniMap />
        <Background gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}
