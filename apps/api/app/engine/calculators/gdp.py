from app.engine.interfaces import IndicatorCalculator, GlobalState, ShockVector

class GDPCalculator(IndicatorCalculator):
    """Calculates GDP impacts from accumulated shocks."""

    def calculate(self, state: GlobalState, applied_shocks: ShockVector) -> GlobalState:
        for country, shocks in applied_shocks.items():
            if country in state:
                # E.g., tariff increases reduce GDP slightly, inflation reduces real GDP
                tariff_shock = shocks.get("tariff", 0.0)
                inflation_increase = shocks.get("inflation_increase", 0.0)
                
                gdp_impact = 0.0
                if tariff_shock > 0:
                    gdp_impact -= tariff_shock * 0.1
                if inflation_increase > 0:
                    gdp_impact -= inflation_increase * 0.2
                
                # Apply the fractional impact
                current_gdp = state[country].get("gdp", 0.0)
                state[country]["gdp"] = current_gdp * (1 + gdp_impact)
                
            elif country == "GLOBAL":
                # Apply global shock to all countries
                global_inflation = shocks.get("inflation_increase", 0.0)
                for c_iso in state.keys():
                    current_gdp = state[c_iso].get("gdp", 0.0)
                    state[c_iso]["gdp"] = current_gdp * (1 - (global_inflation * 0.1))
        
        return state
