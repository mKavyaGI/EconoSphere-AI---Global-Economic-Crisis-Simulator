from app.engine.interfaces import GlobalState, ShockVector

class RecoveryEngine:
    """Applies mean reversion and simulated central bank policy over time."""

    def apply_recovery(self, state: GlobalState, days: int) -> GlobalState:
        """
        Adjusts the global state based on recovery curves.
        If we are past Year 1 (365 days), mean reversion strongly pulls metrics back to baseline.
        """
        import copy
        recovered_state = copy.deepcopy(state)

        if days >= 365:
            # Simple reversion to mean (assuming baseline 0 delta for simplicity in this MVP)
            # In a real model, we would store the true baseline and pull back towards it.
            for country, indicators in recovered_state.items():
                for metric in ["inflation", "gdp"]:
                    if metric in indicators:
                        # Pull back by 20%
                        val = indicators[metric]
                        if metric == "inflation" and val > 0.02:  # Assuming 2% is normal
                            indicators[metric] = val * 0.8
                        if metric == "gdp":
                            # Assume natural growth resumes
                            indicators[metric] = val * 1.05

        return recovered_state
