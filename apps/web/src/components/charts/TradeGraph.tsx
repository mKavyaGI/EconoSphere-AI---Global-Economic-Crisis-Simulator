'use client';

import React, { useMemo } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { useTheme } from 'next-themes';
import { GraphNode, GraphEdge } from '@/hooks/useGraph';

interface TradeGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeSelect?: (node: GraphNode | null) => void;
  filterQuery?: string;
  categoryFilter?: string | null;
}

export default function TradeGraph({ nodes, edges, onNodeSelect, filterQuery, categoryFilter }: TradeGraphProps) {
  const { resolvedTheme } = useTheme();
  
  const isDark = resolvedTheme === 'dark';

  const elements = useMemo(() => {
    const hasActiveFilter = Boolean(filterQuery || categoryFilter);
    const visibleNodeIds = new Set<string>();

    const cyNodes = nodes.map(node => {
      const label = String(node.properties.name || node.id);
      const matchesSearch = !filterQuery || label.toLowerCase().includes(filterQuery.toLowerCase()) || node.id.toLowerCase().includes(filterQuery.toLowerCase());
      const matchesCategory = !categoryFilter || node.label.toLowerCase() === categoryFilter.toLowerCase();
      const isVisible = !hasActiveFilter || (matchesSearch && matchesCategory);

      if (isVisible) {
        visibleNodeIds.add(node.id);
      }

      return {
        data: { 
          id: node.id, 
          label: label,
          category: node.label,
          dimmed: !isVisible ? 'true' : 'false',
          ...node.properties 
        }
      };
    });
    
    const cyEdges = edges.map(edge => {
      const edgeDimmed = !hasActiveFilter || (!visibleNodeIds.has(edge.source) && !visibleNodeIds.has(edge.target));
      return {
        data: { 
          id: edge.id, 
          source: edge.source, 
          target: edge.target,
          label: edge.type,
          dimmed: edgeDimmed ? 'true' : 'false',
          ...edge.properties 
        }
      };
    });

    return [...cyNodes, ...cyEdges];
  }, [nodes, edges, filterQuery, categoryFilter]);

  const stylesheet = [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-y': 5,
        'color': isDark ? '#e5e7eb' : '#374151',
        'font-size': '12px',
        'font-family': 'Inter, sans-serif',
        'background-color': '#3b82f6',
        'width': 30,
        'height': 30,
        'border-width': 2,
        'border-color': isDark ? '#1e3a8a' : '#bfdbfe',
        'transition-property': 'opacity',
        'transition-duration': 0.2
      }
    },
    {
      selector: 'node[category = "Country"]',
      style: {
        'background-color': '#10b981', // Emerald
        'shape': 'ellipse',
      }
    },
    {
      selector: 'node[category = "Commodity"]',
      style: {
        'background-color': '#f59e0b', // Amber
        'shape': 'hexagon',
      }
    },
    {
      selector: 'node[category = "EconomicBloc"]',
      style: {
        'background-color': '#8b5cf6', // Violet
        'shape': 'round-rectangle',
      }
    },
    {
      selector: 'node[dimmed = "true"]',
      style: {
        'opacity': 0.15,
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': isDark ? '#4b5563' : '#9ca3af',
        'target-arrow-color': isDark ? '#4b5563' : '#9ca3af',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'opacity': 0.6,
        'label': 'data(label)',
        'font-size': '8px',
        'color': isDark ? '#9ca3af' : '#6b7280',
        'text-rotation': 'autorotate',
        'transition-property': 'opacity',
        'transition-duration': 0.2
      }
    },
    {
      selector: 'edge[label = "EXPORTS_TO"]',
      style: {
        'line-color': '#3b82f6',
        'target-arrow-color': '#3b82f6',
      }
    },
    {
      selector: 'edge[label = "MEMBER_OF"]',
      style: {
        'line-color': '#8b5cf6',
        'target-arrow-color': '#8b5cf6',
        'line-style': 'dashed'
      }
    },
    {
      selector: 'edge[dimmed = "true"]',
      style: {
        'opacity': 0.1,
      }
    }
  ];

  return (
    <div className="relative w-full h-full bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden shadow-sm">
      {/* Visual Legend */}
      <div className="absolute top-4 left-4 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur border border-gray-200 dark:border-gray-800 rounded-lg p-3 text-xs shadow-sm flex flex-col gap-2">
        <span className="font-semibold text-gray-700 dark:text-gray-200 mb-0.5">Network Nodes</span>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" />
          <span className="text-gray-600 dark:text-gray-300">Country</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-amber-500 inline-block rotate-45" />
          <span className="text-gray-600 dark:text-gray-300">Commodity</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-2.5 rounded-sm bg-violet-500 inline-block" />
          <span className="text-gray-600 dark:text-gray-300">Economic Bloc</span>
        </div>
      </div>

      <CytoscapeComponent
        elements={elements}
        style={{ width: '100%', height: '100%' }}
        stylesheet={stylesheet as React.ComponentProps<typeof CytoscapeComponent>['stylesheet']}
        layout={{ 
          name: 'cose',
          animate: true,
          randomize: true,
          nodeRepulsion: 400000,
          idealEdgeLength: 100
        }}
        wheelSensitivity={0.2}
        cy={(cy: cytoscape.Core) => {
          cy.off('tap', 'node');
          cy.off('tap');

          cy.on('tap', 'node', (evt) => {
            const nodeData = evt.target.data();
            if (onNodeSelect) {
              const matched = nodes.find(n => n.id === nodeData.id);
              onNodeSelect(matched || null);
            }
          });

          cy.on('tap', (evt) => {
            if (evt.target === cy && onNodeSelect) {
              onNodeSelect(null);
            }
          });
        }}
      />
    </div>
  );
}

