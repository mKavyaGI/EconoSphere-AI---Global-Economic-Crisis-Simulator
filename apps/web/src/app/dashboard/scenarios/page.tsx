'use client';

import React, { useState } from 'react';
import { 
  useScenarios, 
  useCreateScenario, 
  useDuplicateScenario, 
  useDeleteScenario, 
  usePublishScenario, 
  useArchiveScenario, 
  useValidateScenario,
  useCreateEvent,
  ScenarioEvent,
  ScenarioValidationResult 
} from '@/hooks/useScenarios';
import ScenarioCanvas from '@/components/scenarios/ScenarioCanvas';
import ScenarioProperties from '@/components/scenarios/ScenarioProperties';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Modal } from '@/components/ui/modal';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import { 
  Loader2, Plus, Copy, Trash2, ShieldCheck, CheckCircle2, 
  AlertTriangle, AlertCircle, Upload, Archive, Layers 
} from 'lucide-react';

export default function ScenariosStudioPage() {
  const { data: scenarios, isLoading, isError, error, refetch } = useScenarios();
  const [selectedScenarioId, setSelectedScenarioId] = useState<number | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<ScenarioEvent | null>(null);
  const [validationResult, setValidationResult] = useState<ScenarioValidationResult | null>(null);

  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    type: 'create' | 'duplicate' | 'delete' | null;
    inputValue: string;
  }>({ isOpen: false, type: null, inputValue: '' });

  const createScenario = useCreateScenario();
  const duplicateScenario = useDuplicateScenario();
  const deleteScenario = useDeleteScenario();
  const publishScenario = usePublishScenario();
  const archiveScenario = useArchiveScenario();
  const validateScenario = useValidateScenario();
  const createEvent = useCreateEvent();

  const selectedScenario = scenarios?.find(s => s.id === selectedScenarioId);
  const latestVersion = selectedScenario?.versions?.[0];
  const events = latestVersion?.events || [];

  const handleCreateScenario = () => {
    setModalState({ isOpen: true, type: 'create', inputValue: 'Global Tariff Shock 2026' });
  };

  const handleDuplicate = () => {
    if (!selectedScenario) return;
    setModalState({ isOpen: true, type: 'duplicate', inputValue: `Copy of ${selectedScenario.title}` });
  };

  const handleDelete = () => {
    if (!selectedScenario) return;
    setModalState({ isOpen: true, type: 'delete', inputValue: '' });
  };

  const handleConfirmModal = () => {
    if (modalState.type === 'create') {
      if (!modalState.inputValue) return;
      createScenario.mutate(
        { title: modalState.inputValue, description: 'Custom causal shock experiment created via studio.', is_template: false, tags: ['custom', 'macro'] },
        {
          onSuccess: (newScenario) => {
            setSelectedScenarioId(newScenario.id);
            setSelectedEvent(null);
            setValidationResult(null);
            setModalState({ isOpen: false, type: null, inputValue: '' });
          }
        }
      );
    } else if (modalState.type === 'duplicate') {
      if (!selectedScenario || !modalState.inputValue) return;
      duplicateScenario.mutate(
        { id: selectedScenario.id, newTitle: modalState.inputValue },
        {
          onSuccess: (newScenario) => {
            setSelectedScenarioId(newScenario.id);
            setSelectedEvent(null);
            setValidationResult(null);
            setModalState({ isOpen: false, type: null, inputValue: '' });
          }
        }
      );
    } else if (modalState.type === 'delete') {
      if (!selectedScenario) return;
      deleteScenario.mutate(selectedScenario.id, {
        onSuccess: () => {
          setSelectedScenarioId(null);
          setSelectedEvent(null);
          setValidationResult(null);
          setModalState({ isOpen: false, type: null, inputValue: '' });
        }
      });
    }
  };

  const handleValidate = () => {
    if (!selectedScenarioId) return;
    validateScenario.mutate(selectedScenarioId, {
      onSuccess: (result) => setValidationResult(result)
    });
  };

  const handleAddEvent = () => {
    if (!selectedScenarioId) return;
    createEvent.mutate(
      {
        scenarioId: selectedScenarioId,
        data: {
          title: selectedEvent ? `Sub-shock of ${selectedEvent.title}` : 'Primary Macro Shock',
          category: 'ECONOMIC',
          status: 'PENDING',
          parameters: {},
          shockIntensity: 0.5,
          propagationDelay: selectedEvent ? (selectedEvent.propagationDelay || 0) + 10 : 0,
          parent_event_id: selectedEvent ? selectedEvent.id : null,
        }
      },
      {
        onSuccess: (newEv) => {
          setSelectedEvent(newEv);
          setValidationResult(null);
        }
      }
    );
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] p-4 space-y-4 max-w-7xl mx-auto w-full">
      
      {/* Top Title & Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-4 shadow-sm">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white flex items-center gap-2">
            <Layers className="w-6 h-6 text-blue-500" />
            Scenario Studio
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">Design, validate, and simulate causal macro-economic shockwaves</p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <div className="w-72">
            {isLoading ? (
              <div className="flex items-center text-xs text-muted-foreground bg-gray-50 dark:bg-gray-950 p-2 rounded border"><Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-500" /> Fetching scenarios index...</div>
            ) : (
              <Select 
                value={selectedScenarioId?.toString() || ''} 
                onValueChange={(val) => {
                  setSelectedScenarioId(parseInt(val, 10));
                  setSelectedEvent(null);
                  setValidationResult(null);
                }}
              >
                <SelectTrigger className="bg-gray-50 dark:bg-gray-950 text-sm h-10">
                  <SelectValue placeholder="Select a scenario to edit..." />
                </SelectTrigger>
                <SelectContent>
                  {scenarios?.map((scenario) => (
                    <SelectItem key={scenario.id} value={scenario.id.toString()} className="text-sm">
                      <span className="font-semibold">{scenario.title}</span>
                      <span className="text-[10px] ml-2 px-1.5 py-0.5 rounded uppercase font-mono bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
                        {scenario.status || 'DRAFT'}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          <Button 
            onClick={handleCreateScenario} 
            disabled={createScenario.isPending} 
            className="flex items-center gap-1.5 text-sm h-10 cursor-pointer shadow-sm"
          >
            <Plus className="w-4 h-4" />
            {createScenario.isPending ? 'Creating...' : 'New Scenario'}
          </Button>
        </div>
      </div>

      {/* Selected Scenario Lifecycle Actions Bar */}
      {selectedScenario && (
        <div className="bg-gray-50/80 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-gray-700 dark:text-gray-300">Active Blueprint:</span>
            <span className="px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300 font-bold border border-blue-200 dark:border-blue-900">
              {selectedScenario.title}
            </span>
            <span className="px-2 py-0.5 rounded bg-gray-200 dark:bg-gray-800 font-mono text-[10px] uppercase">
              {selectedScenario.status || 'DRAFT'}
            </span>
            <span className="text-gray-400 dark:text-gray-600">|</span>
            <span className="text-gray-500">Events: <strong>{events.length}</strong></span>
          </div>

          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleAddEvent}
              disabled={createEvent.isPending}
              className="text-xs h-8 bg-white dark:bg-gray-900 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800 hover:bg-emerald-50 dark:hover:bg-emerald-950/40 cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5 mr-1" />
              {selectedEvent ? 'Add Child Event' : 'Add Root Shock'}
            </Button>

            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleValidate} 
              disabled={validateScenario.isPending}
              className="text-xs h-8 bg-white dark:bg-gray-900 cursor-pointer"
            >
              <ShieldCheck className="w-3.5 h-3.5 mr-1 text-blue-500" />
              {validateScenario.isPending ? 'Checking...' : 'Validate Blueprint'}
            </Button>

            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleDuplicate}
              disabled={duplicateScenario.isPending} 
              className="text-xs h-8 bg-white dark:bg-gray-900 cursor-pointer"
            >
              <Copy className="w-3.5 h-3.5 mr-1 text-gray-500" />
              Duplicate
            </Button>

            {selectedScenario.status !== 'PUBLISHED' ? (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => publishScenario.mutate(selectedScenario.id)}
                disabled={publishScenario.isPending}
                className="text-xs h-8 bg-white dark:bg-gray-900 text-violet-600 dark:text-violet-400 border-violet-200 dark:border-violet-900 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5 mr-1" />
                Publish
              </Button>
            ) : (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => archiveScenario.mutate(selectedScenario.id)}
                disabled={archiveScenario.isPending}
                className="text-xs h-8 bg-white dark:bg-gray-900 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-900 cursor-pointer"
              >
                <Archive className="w-3.5 h-3.5 mr-1" />
                Archive
              </Button>
            )}

            <Button 
              variant="ghost" 
              size="sm" 
              onClick={handleDelete}
              disabled={deleteScenario.isPending}
              className="text-xs h-8 text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-950/30 px-2 cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      )}

      {/* Real-time Validation Feedback Banner */}
      {validationResult && (
        <div className={`p-4 rounded-xl border ${validationResult.is_valid ? 'bg-emerald-50/80 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900' : 'bg-red-50/80 dark:bg-red-950/30 border-red-200 dark:border-red-900'} flex flex-col gap-2 transition-all`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-bold text-sm">
              {validationResult.is_valid ? (
                <>
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-emerald-900 dark:text-emerald-300">Scenario Passed Structural Validation</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                  <span className="text-red-900 dark:text-red-300">Scenario Failed Validation ({validationResult.errors.length} errors)</span>
                </>
              )}
            </div>
            <button onClick={() => setValidationResult(null)} className="text-xs text-gray-400 hover:text-gray-600 underline">Dismiss</button>
          </div>

          {validationResult.errors.length > 0 && (
            <ul className="list-disc list-inside text-xs text-red-700 dark:text-red-400 space-y-0.5 pl-2">
              {validationResult.errors.map((err, idx) => <li key={idx} className="font-mono">{err}</li>)}
            </ul>
          )}

          {validationResult.warnings.length > 0 && (
            <div className="mt-1 flex flex-col gap-1">
              {validationResult.warnings.map((warn, idx) => (
                <div key={idx} className="flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400 bg-amber-100/50 dark:bg-amber-950/30 px-2.5 py-1 rounded border border-amber-200/60 dark:border-amber-900/40">
                  <AlertTriangle className="w-3.5 h-3.5 flex-none" />
                  <span>{warn}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Main Studio Work Area */}
      <div className="flex-1 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden flex bg-white dark:bg-gray-900 shadow-sm min-h-[550px]">
        {/* Main Canvas Area */}
        <div className="flex-1 relative min-h-[550px]">
          {isError && (
            <div className="p-6 h-full flex items-center justify-center">
              <ErrorState 
                title="Could Not Load Scenarios Index"
                error={error}
                onRetry={refetch}
              />
            </div>
          )}

          {!isLoading && !isError && (!scenarios || scenarios.length === 0) && (
            <div className="p-6 h-full flex items-center justify-center">
              <EmptyState
                title="No Scenarios Available"
                description="Your workspace currently has no simulation scenarios or macroeconomic shock templates."
                action={{ label: 'Create Initial Scenario', onClick: handleCreateScenario }}
              />
            </div>
          )}

          {!isLoading && !isError && scenarios && scenarios.length > 0 && !selectedScenarioId && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground p-6 text-center bg-gray-50/50 dark:bg-gray-950/50">
              <Layers className="w-12 h-12 text-blue-500 opacity-80 mb-3 animate-pulse" />
              <h3 className="text-base font-bold text-gray-900 dark:text-white mb-1">Select a Blueprint to Begin</h3>
              <p className="text-sm max-w-sm">Choose a scenario from the dropdown above or create a new simulation blueprint to start inspecting causal trees.</p>
            </div>
          )}

          {selectedScenarioId && (
            <ScenarioCanvas 
              events={events} 
              onEventSelect={setSelectedEvent} 
            />
          )}
        </div>

        {/* Right Sidebar */}
        {selectedScenarioId && (
          <div className="w-80 border-l border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm">
            <ScenarioProperties 
              event={selectedEvent} 
              scenarioId={selectedScenarioId} 
              onDeleted={() => setSelectedEvent(null)}
            />
          </div>
        )}
      </div>

      {/* Action Modals */}
      <Modal
        isOpen={modalState.isOpen}
        onClose={() => setModalState({ isOpen: false, type: null, inputValue: '' })}
        title={
          modalState.type === 'create' ? 'New Scenario Blueprint' :
          modalState.type === 'duplicate' ? 'Duplicate Scenario' :
          modalState.type === 'delete' ? 'Delete Scenario' : ''
        }
        description={
          modalState.type === 'delete' 
            ? `Are you sure you want to permanently delete scenario "${selectedScenario?.title}"? This action cannot be undone.`
            : ''
        }
        footer={
          <>
            <Button variant="ghost" onClick={() => setModalState({ isOpen: false, type: null, inputValue: '' })}>
              Cancel
            </Button>
            <Button 
              variant={modalState.type === 'delete' ? 'destructive' : 'default'} 
              onClick={handleConfirmModal}
              disabled={createScenario.isPending || duplicateScenario.isPending || deleteScenario.isPending}
            >
              {modalState.type === 'delete' ? 'Delete Permanently' : 'Confirm'}
            </Button>
          </>
        }
      >
        {modalState.type !== 'delete' && (
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Scenario Title
              </label>
              <Input
                value={modalState.inputValue}
                onChange={(e) => setModalState({ ...modalState, inputValue: e.target.value })}
                placeholder="Enter scenario title..."
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleConfirmModal();
                }}
              />
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

