import React from 'react';
import { Play, Pause, FastForward, Rewind } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface TimelinePlayerProps {
  currentHorizon: number;
  isPlaying: boolean;
  onPlayPause: () => void;
}

export function TimelinePlayer({ currentHorizon, isPlaying, onPlayPause }: TimelinePlayerProps) {
  return (
    <div className="flex items-center space-x-4 bg-card border rounded-lg p-2 px-4">
      <div className="flex items-center space-x-1">
        <Button variant="ghost" size="icon">
          <Rewind className="w-4 h-4" />
        </Button>
        <Button variant="default" size="icon" onClick={onPlayPause}>
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>
        <Button variant="ghost" size="icon">
          <FastForward className="w-4 h-4" />
        </Button>
      </div>
      <div className="h-8 border-l" />
      <div className="flex flex-col">
        <span className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Time Horizon</span>
        <span className="font-mono text-sm">Day {currentHorizon}</span>
      </div>
    </div>
  );
}
