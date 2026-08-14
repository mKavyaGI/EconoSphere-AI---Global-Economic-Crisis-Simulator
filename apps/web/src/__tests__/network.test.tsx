import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorState } from '@/components/ui/ErrorState';

describe('Trade Network & UI Architecture Unit Tests', () => {
  it('renders EmptyState component with correct title, description, and action', () => {
    const handleClick = vi.fn();
    render(
      <EmptyState
        title="No Trade Routes Found"
        description="There are no trade links exceeding the specified minimum volume threshold."
        action={{ label: 'Reset Filter', onClick: handleClick }}
      />
    );

    expect(screen.getByText('No Trade Routes Found')).toBeDefined();
    expect(screen.getByText('There are no trade links exceeding the specified minimum volume threshold.')).toBeDefined();

    const button = screen.getByText('Reset Filter');
    expect(button).toBeDefined();
    fireEvent.click(button);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('renders ErrorState component with error details and retry functionality', () => {
    const handleRetry = vi.fn();
    const testError = new Error('Network timeout fetching Neo4j subgraph');
    
    render(
      <ErrorState
        title="Failed to Load Trade Graph"
        error={testError}
        onRetry={handleRetry}
      />
    );

    expect(screen.getByText('Failed to Load Trade Graph')).toBeDefined();
    expect(screen.getByText('Network timeout fetching Neo4j subgraph')).toBeDefined();

    const retryButton = screen.getByText('Try Again');
    expect(retryButton).toBeDefined();
    fireEvent.click(retryButton);
    expect(handleRetry).toHaveBeenCalledTimes(1);
  });
});
