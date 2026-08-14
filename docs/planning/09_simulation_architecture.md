# 09. Simulation Architecture Blueprint

**Module:** Global Economic Simulation Engine
**Status:** Approved for Implementation
**Date:** 2026-07-23

---

## 1. Executive Summary
The Global Economic Simulation Engine is the core orchestration layer of EconoSphere AI. It transitions static scenario definitions into dynamic, cascading economic models over discrete time horizons. 

To ensure long-term extensibility, the engine is **configuration-driven** and utilizes the **Strategy Pattern** for shock propagation. It operates asynchronously via a Redis-backed message broker and streams real-time state snapshots to the client via WebSockets.

---

## 2. Simulation Lifecycle

1. **Initialization:** The user submits a scenario via the frontend. FastAPI validates the request and publishes a `SimulationTask` to the Redis Queue.
2. **Compilation:** The background worker pulls the task. The `ScenarioCompiler` fetches baseline data from PostgreSQL (indicators) and Neo4j (trade network).
3. **Execution Loop (Per Time Horizon):**
   - **Event Processing:** Initial shocks are injected based on the scenario's events.
   - **Strategy Resolution:** The `StrategyManager` invokes relevant domain strategies (Trade, Currency, Energy) to propagate the shock.
   - **Calculation:** Independent `IndicatorCalculators` compute the net change on macroeconomic KPIs (GDP, Inflation).
   - **Recovery & Decay:** The `RecoveryEngine` applies mean-reversion and central bank policy adjustments.
   - **Snapshotting:** The exact state of the world is persisted as a `SimulationSnapshot`.
4. **Streaming:** As each snapshot is generated, it is emitted over the Redis pub/sub to the FastAPI WebSocket server, which broadcasts it to the client.
5. **Finalization:** The simulation completes. An `AIContextBuilder` payload is prepared for future policy recommendations.

---

## 3. Pipeline Architecture

The pipeline strictly isolates responsibilities to avoid a monolithic orchestrator.

```text
Simulation Engine
       │
       ▼
Scenario Compiler (Fetches baseline data & Config)
       │
       ▼
Event Processor (Parses raw ScenarioEvents into quantitative Shocks)
       │
       ▼
Dependency Resolver (Queries Neo4j up to `max_depth` hops)
       │
       ▼
Strategy Manager (Orchestrator for Domain Strategies)
       │
 ┌─────┼────────┬────────┬─────────┐
 ▼     ▼        ▼        ▼         ▼
Trade Banking  Energy  Currency  Supply Chain
       │
       ▼
Indicator Calculators (GDP, Inflation, Employment)
       │
       ▼
Timeline Generator (Iterates through Configured Horizons: Day 1, Week 1...)
       │
       ▼
Recovery Engine (Applies decay and reversion functions)
       │
       ▼
Snapshot Store (Persists state to PostgreSQL)
       │
       ▼
Result API & WebSocket Stream
```

---

## 4. Configuration Model

The engine behavior is governed by `SimulationConfig`, avoiding hardcoded assumptions.

```json
{
  "time_horizons": [1, 7, 30, 180, 365, 1095],
  "max_depth": 3,
  "propagation_threshold": 0.05,
  "decay_factor": 0.1,
  "random_seed": 42,
  "confidence_level": 0.95,
  "max_iterations": 10
}
```

---

## 5. Extensibility: Strategy Interfaces

The system uses abstract base classes to allow swapping or adding economic domains.

```python
class SimulationStrategy(ABC):
    @abstractmethod
    def apply_shock(self, state: GlobalState, shock: ShockVector) -> GlobalState:
        pass

class TradeStrategy(SimulationStrategy):
    def apply_shock(self, state, shock):
        # Implementation for trade cascades using Neo4j dependencies
        pass
```
New domains (e.g., Climate, Banking) only require subclassing `SimulationStrategy` and registering it with the `StrategyManager`.

---

## 6. Snapshot Model

Instead of only storing the end result, the engine stores a `SimulationSnapshot` for each time horizon, enabling **Replay functionality**.

- `run_id` (UUID)
- `horizon_days` (int: e.g., 30)
- `timestamp` (DateTime)
- `global_state` (JSONB: all indicator values for all countries at this moment)
- `active_shocks` (JSONB: ongoing cascading effects)

---

## 7. Event Logging

A detailed `SimulationLog` is generated to provide transparency into how the engine reached its conclusions.
Fields: `timestamp`, `component` (e.g., 'TradeStrategy'), `message` (e.g., 'China exports reduced by 25%'), `severity`, `affected_country`, `affected_indicator`.

---

## 8. WebSocket Protocol

**Bidirectional Control:**
- **Client $\rightarrow$ Server:** `{"command": "PAUSE"}`, `{"command": "RESUME"}`, `{"command": "JUMP", "horizon": 30}`
- **Server $\rightarrow$ Client:** `{"type": "SNAPSHOT", "data": {...}}`, `{"type": "LOG", "message": "..."}`

---

## 9. Performance Considerations

- **Redis Caching:** Neo4j graph topologies are cached in Redis to prevent identical recursive queries during standard cascades.
- **Asynchronous Workers:** Compute-heavy matrix multiplications and loop iterations occur off the main FastAPI ASGI loop in a dedicated Worker (e.g., RQ).
- **JSONB Aggregation:** The `global_state` is stored as JSONB to prevent thousands of granular INSERTs into a relational table per snapshot.
