# Graph Architecture Document

## 1. Vision: Economic Relationship Graph
EconoSphere AI leverages Neo4j to model not just trade, but the complete macro-economic topology of the world. This allows for rich analytics, community detection, and powers the Phase 9 Simulation Engine.

## 2. Node Definitions

### Core Entities
- **`Country`**
  - `iso3` (String) - Primary Key
  - `name` (String)
  - `region` (String)
  - `income_group` (String)
  - *Simulation Hooks:*
    - `health` (Float) - Overall economic health (0.0 to 1.0)
    - `stability` (Float) - Political/Economic stability (0.0 to 1.0)
    - `risk` (Float) - Current risk exposure (0.0 to 1.0)
    - `shockValue` (Float) - Accumulator for crisis propagation

- **`Commodity`**
  - `id` (String) - Primary Key
  - `name` (String)
  - `category` (String)

- **`EconomicBloc`**
  - `id` (String) - Primary Key
  - `name` (String)
  - `type` (String) - Trade, Political, etc.

## 3. Relationship Definitions

Relationships are directional, weighted, and support time-series data to enable historical replays.

### Trade Relationships
- **`EXPORTS_TO`** (Country -> Country)
  - `commodity_id` (String)
  - `year` (Integer)
  - `tradeVolume` (Float)
  - `growth` (Float)
  - `dependencyScore` (Float)

- **`IMPORTS_FROM`** (Country -> Country)
  - *Inverse of EXPORTS_TO, typically implied by edge direction, but explicit edges can aid performance.*

### Future-Proofed Relationships
- **`INVESTS_IN`** (Country -> Country)
  - `volume` (Float), `sector` (String)
- **`SANCTIONS`** (Country -> Country)
  - `severity` (Float), `active` (Boolean)
- **`MEMBER_OF`** (Country -> EconomicBloc)
  - `joined_year` (Integer)

## 4. Graph Algorithms & Cypher Query Patterns

### Generic Queries
- **Pathfinding**: Shortest path between two nodes (e.g., assessing supply chain vulnerability).
  ```cypher
  MATCH p=shortestPath((c1:Country {iso3: $iso1})-[:EXPORTS_TO*..5]->(c2:Country {iso3: $iso2}))
  RETURN p
  ```
- **Neighbors**: 1-hop or 2-hop dependencies.
- **Subgraph**: Extract the subgraph for a specific commodity (e.g., only "Oil" relationships).

### Algorithms (via GDS library or manual Cypher)
- **PageRank**: Determines the most globally influential trading nation.
- **Betweenness Centrality**: Identifies critical supply chain hubs (choke points).
- **Community Detection**: Finds highly intertwined economic clusters (e.g., using Louvain).

## 5. Performance & Index Strategy
- Create unique constraints on `Country(iso3)`, `Commodity(id)`, `EconomicBloc(id)`.
- Create indexes on `EXPORTS_TO(year)` and `EXPORTS_TO(commodity_id)` to quickly filter time-series edges.
- Cache heavily queried static subgraphs (e.g., base network) using Redis.
- Paginate neighbors and graph API responses to prevent massive JSON payloads.

## 6. Simulation Engine Hooks
Nodes contain `shockValue` properties. In the future, a Python service will trigger an event (e.g., "Suez Canal Blocked"), adjust the `shockValue` on affected nodes, and propagate it through the graph based on the `dependencyScore` weight of the edges.
