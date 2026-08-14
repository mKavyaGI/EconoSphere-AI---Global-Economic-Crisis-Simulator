"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Play, Pause, FastForward, Rewind } from "lucide-react";
import { apiClient } from "@/lib/api/client";
import { ImpactMap } from "@/components/simulation/ImpactMap";

/**
 * M-2 Fix: Derive WebSocket URL from the shared API base URL.
 * Converts http:// → ws:// and https:// → wss:// for TLS-compatible production.
 */
function getWsBase(): string {
  const apiBase =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
  return apiBase.replace(/^http/, "ws");
}

export default function SimulationDashboard() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentHorizon, setCurrentHorizon] = useState(0);
  const [runId, setRunId] = useState<number | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [globalState, setGlobalState] = useState<Record<string, any> | undefined>(undefined);
  const [activeShocks, setActiveShocks] = useState<Record<string, any> | undefined>(undefined);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (runId) {
      // M-2 Fix: Construct WebSocket URL from env variable (wss:// in production)
      ws.current = new WebSocket(`${getWsBase()}/simulation/${runId}/stream`);
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "SNAPSHOT") {
          setCurrentHorizon(data.horizon);
          setLogs(prev => [...prev, data.message]);
          if (data.global_state) setGlobalState(data.global_state);
          if (data.active_shocks) setActiveShocks(data.active_shocks);
        }
      };

      return () => {
        ws.current?.close();
      };
    }
  }, [runId]);

  const startSimulation = async () => {
    try {
      // M-2 Fix: Use shared apiClient (reads NEXT_PUBLIC_API_URL)
      const { data } = await apiClient.post("/simulation/start", {
        scenario_id: 1,
        config: {
          time_horizons: [1, 7, 30, 180, 365, 1095],
          max_depth: 3
        }
      });
      setRunId(data.id);
      setIsPlaying(true);
    } catch (err) {
      console.error(err);
    }
  };


  const handlePlayPause = () => {
    if (!ws.current) return;
    const command = isPlaying ? "PAUSE" : "RESUME";
    ws.current.send(JSON.stringify({ command }));
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="flex flex-col h-full space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Simulation Command Center</h1>
        
        <div className="flex items-center space-x-2 bg-card border rounded-lg p-1">
          <Button variant="ghost" size="icon" disabled={!runId}>
            <Rewind className="w-4 h-4" />
          </Button>
          <Button 
            variant="default" 
            size="icon" 
            onClick={runId ? handlePlayPause : startSimulation}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <Button variant="ghost" size="icon" disabled={!runId}>
            <FastForward className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 flex-1">
        <div className="col-span-2 bg-card border rounded-xl overflow-hidden flex items-center justify-center relative">
          <div className="absolute top-4 right-4 z-20 bg-background/80 backdrop-blur px-3 py-1 rounded-full border text-sm font-medium">
            Day {currentHorizon}
          </div>
          <ImpactMap currentHorizon={currentHorizon} isRunning={isPlaying} globalState={globalState} activeShocks={activeShocks} />
        </div>
        
        <div className="bg-card border rounded-xl p-4 flex flex-col space-y-4">
          <h3 className="font-semibold text-lg">Event Log</h3>
          <div className="flex-1 overflow-y-auto space-y-2 text-sm text-muted-foreground font-mono">
            {logs.length === 0 && <div>Ready to start simulation...</div>}
            {logs.map((log, i) => (
              <div key={i} className="border-l-2 border-primary pl-2">
                {log}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
