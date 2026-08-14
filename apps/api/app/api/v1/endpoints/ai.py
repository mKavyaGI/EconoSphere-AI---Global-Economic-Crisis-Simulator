"""
FastAPI route endpoints for Phase 10: AI Forecasting & Decision Intelligence.

Provides enterprise REST APIs for:
- Multi-horizon macroeconomic predictive forecasts with confidence bounds and SHAP explainability.
- Actionable prescriptive policy recommendations with simulated trade-offs and confidence scores.
- Multi-dimensional sovereign risk assessment scorecards (0-100 scale, Low-Critical tiers).
- Unsupervised global statistical and trade disruption anomaly detection.
- Model Registry inspection and distributed RQ worker task dispatch.
"""
from fastapi import APIRouter, Query, HTTPException, status
from typing import List, Optional, Any, Dict
from app.ai.services.orchestrator import AIDecisionService
from app.ai.schemas import (
    CountryForecastResponse,
    CountryRiskScorecard,
    PolicyRecommendation,
    AnomalyAlert,
    ModelMetadata
)
from app.ai.registry.manager import model_registry

router = APIRouter()

@router.get(
    "/forecast/{iso3}",
    response_model=CountryForecastResponse,
    summary="Get Multi-Horizon AI Macroeconomic Forecasts",
    description="Returns multi-horizon projections across 8 indicators complete with P10/P50/P90 confidence intervals, SHAP feature attributions, and Natural Language explanations."
)
async def get_sovereign_forecast(
    iso3: str,
    model_type: str = Query("XGBOOST", description="Algorithm architecture: XGBOOST, LIGHTGBM, TRANSFORMER, GNN, VAR, RL"),
    force_refresh: bool = Query(False, description="Bypass 1-hour Redis caching to compute fresh inferences")
):
    service = AIDecisionService(model_type=model_type)
    try:
        return await service.get_country_forecast(iso3.upper(), force_refresh=force_refresh)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AI Forecasting failed: {str(e)}")

@router.get(
    "/risk/{iso3}",
    response_model=CountryRiskScorecard,
    summary="Get Sovereign Structural Risk Assessment Scorecard",
    description="Computes normalized 0-100 hazard scores and Low/Medium/High/Critical tier rankings across 7 structural macroeconomic dimensions."
)
async def get_sovereign_risk_scorecard(
    iso3: str,
    force_refresh: bool = Query(False)
):
    service = AIDecisionService()
    try:
        return await service.get_risk_assessment(iso3.upper(), force_refresh=force_refresh)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Risk engine execution failed: {str(e)}")

@router.get(
    "/recommendations/{iso3}",
    response_model=List[PolicyRecommendation],
    summary="Get AI Actionable Policy Recommendations & Simulated Trade-offs",
    description="Evaluates country risk scorecards and simulates explicit macroeconomic trade-offs for proposed strategic interventions."
)
async def get_policy_recommendations(
    iso3: str,
    sim_run_id: Optional[str] = Query(None, description="Optional simulation scenario context ID")
):
    service = AIDecisionService()
    try:
        return await service.get_policy_recommendations(iso3.upper(), sim_run_id=sim_run_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Recommendation engine failed: {str(e)}")

@router.get(
    "/anomalies",
    response_model=List[AnomalyAlert],
    summary="Scan Global Economic System for Statistical & Trade Anomalies",
    description="Scans current multi-country indicators to detect severe statistical anomalies and Neo4j supply corridor disruptions."
)
async def scan_anomalies():
    service = AIDecisionService()
    try:
        return await service.scan_global_anomalies()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Anomaly scanner failed: {str(e)}")

@router.get(
    "/models",
    response_model=List[ModelMetadata],
    summary="List Registered AI Predictive Models & Validation KPIs",
    description="Inspects Model Registry lineages, hyperparameter defaults, and RMSE/R² k-fold evaluation metrics."
)
async def list_registered_models():
    return model_registry.list_models()

@router.post(
    "/forecast/{iso3}/async",
    summary="Dispatch Asynchronous AI Batch Forecast Task",
    description="Queues long-running deep learning projections to background Redis RQ worker nodes."
)
async def dispatch_async_forecast(
    iso3: str,
    model_type: str = Query("XGBOOST")
):
    service = AIDecisionService(model_type=model_type)
    return service.trigger_async_batch_forecast(iso3.upper(), model_type=model_type)
