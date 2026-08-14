'use client';

import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { ScenarioEvent, useUpdateEvent, useDeleteEvent } from '@/hooks/useScenarios';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Trash2, Save, Activity } from 'lucide-react';

const eventSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255, 'Title too long'),
  shockIntensity: z
    .number({ error: 'Must be a number' })
    .min(0, 'Must be between 0 and 1')
    .max(1, 'Must be between 0 and 1'),
  propagationDelay: z
    .number({ error: 'Must be a number' })
    .int('Must be a whole number')
    .min(0, 'Cannot be negative'),
});

type EventFormValues = z.infer<typeof eventSchema>;

interface ScenarioPropertiesProps {
  event: ScenarioEvent | null;
  scenarioId: number;
  onDeleted?: () => void;
}

export default function ScenarioProperties({ event, scenarioId, onDeleted }: ScenarioPropertiesProps) {
  const { mutate: updateEvent, isPending } = useUpdateEvent();
  const { mutate: deleteEvent, isPending: isDeleting } = useDeleteEvent();
  
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<EventFormValues>({
    resolver: zodResolver(eventSchema),
    defaultValues: {
      title: '',
      shockIntensity: 0,
      propagationDelay: 0,
    },
  });

  // Reset form when event changes
  React.useEffect(() => {
    if (event) {
      reset({
        title: event.title,
        shockIntensity: event.shockIntensity ?? 0,
        propagationDelay: event.propagationDelay ?? 0,
      });
    }
  }, [event, reset]);

  if (!event) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground p-6 text-center">
        <div>
          <Activity className="w-8 h-8 mx-auto mb-2 text-gray-400 dark:text-gray-600 opacity-60" />
          <p className="font-medium mb-1 text-sm text-gray-700 dark:text-gray-300">No Event Selected</p>
          <p className="text-xs text-muted-foreground max-w-[200px]">Click a node on the causal canvas to view and edit its parameters.</p>
        </div>
      </div>
    );
  }

  const onSubmit = (data: EventFormValues) => {
    updateEvent({
      scenarioId,
      eventId: event.id,
      data: {
        title: data.title,
        shockIntensity: data.shockIntensity,
        propagationDelay: data.propagationDelay,
      },
    });
  };

  const handleDelete = () => {
    if (window.confirm(`Are you sure you want to delete event '${event.title}'?`)) {
      deleteEvent({ scenarioId, eventId: event.id }, {
        onSuccess: () => {
          if (onDeleted) onDeleted();
        }
      });
    }
  };

  return (
    <div className="p-4 flex flex-col h-full overflow-y-auto">
      <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-2.5 mb-4">
        <div>
          <h3 className="text-base font-bold text-gray-900 dark:text-white">Event Parameters</h3>
          <p className="text-[11px] text-gray-400 font-mono">ID: {event.id} · {event.category}</p>
        </div>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 flex-1 flex flex-col">
        {/* Title */}
        <div className="space-y-1">
          <Label htmlFor="title" className="text-xs font-semibold">Event Title</Label>
          <Input id="title" {...register('title')} className="text-sm h-9" />
          {errors.title && (
            <p className="text-xs text-destructive">{errors.title.message}</p>
          )}
        </div>

        {/* Category (read-only) */}
        <div className="space-y-1">
          <Label className="text-xs font-semibold">Category Domain</Label>
          <div className="text-xs px-3 py-2 bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 rounded-md font-mono text-gray-700 dark:text-gray-300">
            {event.category}
          </div>
        </div>

        {/* Shock Intensity */}
        <div className="space-y-1">
          <Label htmlFor="shockIntensity" className="text-xs font-semibold">Shock Intensity (0.0 – 1.0)</Label>
          <Input
            id="shockIntensity"
            type="number"
            step="0.01"
            min="0"
            max="1"
            {...register('shockIntensity', { valueAsNumber: true })}
            className="text-sm h-9 font-mono"
          />
          {errors.shockIntensity && (
            <p className="text-xs text-destructive">{errors.shockIntensity.message}</p>
          )}
        </div>

        {/* Propagation Delay */}
        <div className="space-y-1">
          <Label htmlFor="propagationDelay" className="text-xs font-semibold">Propagation Delay (Days)</Label>
          <Input
            id="propagationDelay"
            type="number"
            min="0"
            {...register('propagationDelay', { valueAsNumber: true })}
            className="text-sm h-9 font-mono"
          />
          {errors.propagationDelay && (
            <p className="text-xs text-destructive">{errors.propagationDelay.message}</p>
          )}
        </div>

        {/* Active Simulation Preview */}
        <div className="mt-4 p-3 border rounded-lg bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/50 space-y-1">
          <h4 className="text-xs font-bold text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Live Engine Sync
          </h4>
          <p className="text-[11px] text-emerald-700/80 dark:text-emerald-400/80 leading-relaxed">
            Parameter updates instantly recalibrate causal transmission weights across simulation pipelines.
          </p>
        </div>

        <div className="mt-auto pt-4 flex flex-col gap-2 border-t border-gray-100 dark:border-gray-800">
          <Button type="submit" size="sm" className="w-full flex items-center gap-1.5 font-semibold cursor-pointer" disabled={isPending || isDeleting}>
            <Save className="w-3.5 h-3.5" />
            {isPending ? 'Saving…' : 'Save Event Parameters'}
          </Button>
          
          <Button 
            type="button" 
            variant="outline" 
            size="sm" 
            onClick={handleDelete}
            disabled={isPending || isDeleting}
            className="w-full text-red-600 dark:text-red-400 border-red-200 dark:border-red-900 hover:bg-red-50 dark:hover:bg-red-950/30 flex items-center gap-1.5 text-xs cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {isDeleting ? 'Deleting…' : 'Delete This Event'}
          </Button>
        </div>
      </form>
    </div>
  );
}
