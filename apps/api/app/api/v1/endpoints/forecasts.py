"""
EconoSphere AI — Phase 11 Step 13 Forecast Endpoints
======================================================
FastAPI router for the locked T+1 GDP growth production forecast.

These endpoints serve the official Phase 11 forecast artifact. They do NOT
retrain or re-invoke the ML model. All values are read from the pre-generated
CSV artifact produced by train_t1_adaptive_uncertainty.py.

Endpoints:
  GET /forecasts                → All available country forecasts
  GET /forecasts/metadata       → Model provenance metadata
  GET /forecasts/{country_code} → Single-country forecast or 404

Phase 13 Step 1 Inference Endpoints:
  GET  /forecasts/model-info    → Frozen Phase 11 model identity
  POST /forecasts/gdp           → Live inference against frozen model

Phase 13 Step 2 Operational Safety Endpoints:
  GET  /forecasts/health        → Production health summary
  GET  /forecasts/readiness     → Production readiness gate

API prefix (from router.py): /api/v1/forecasts

Statistical disclaimer:
  All responses include forecast_status='forecast' to clearly indicate that
  values are model-generated projections, not observed GDP statistics.

Security policy (Step 2):
  FrozenModelVerificationError details (which contain raw hash values and
  absolute paths) are NEVER forwarded to API callers. Internal diagnostics
  are logged server-side only.
"""
import time
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from app.schemas.forecast import (
    GDPForecastResponse,
    ForecastListResponse,
    ForecastMetadataResponse,
)
import app.services.forecast_service as forecast_service
import structlog

router = APIRouter()

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Standard forecast artifact endpoints (unchanged from Step 1)
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=ForecastListResponse,
    summary="Get All Official GDP T+1 Production Forecasts",
    description=(
        "Returns the official Phase 11 2026 GDP growth forecasts for all available countries, "
        "produced by the locked HistGradientBoostingRegressor pipeline (Test RMSE=3.9113). "
        "Includes 90% prediction intervals calibrated from validation residuals (Q90=11.4507). "
        "These are MODEL FORECASTS, not observed GDP statistics."
    ),
)
async def get_all_forecasts() -> ForecastListResponse:
    """Return all available official country GDP growth forecasts."""
    try:
        forecasts_raw = forecast_service.get_all_forecasts()
        metadata_raw = forecast_service.get_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast artifact unavailable. The ML pipeline output has not been generated.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forecast artifact is invalid or corrupted. Contact the system administrator.",
        ) from exc

    forecasts = [GDPForecastResponse(**f) for f in forecasts_raw]
    metadata = ForecastMetadataResponse(**metadata_raw)
    return ForecastListResponse(forecasts=forecasts, metadata=metadata)


@router.get(
    "/metadata",
    response_model=ForecastMetadataResponse,
    summary="Get Official Forecast Model Metadata",
    description=(
        "Returns provenance metadata for the locked Phase 11 ML forecasting pipeline: "
        "model version, algorithm, test RMSE, training/validation/test periods, "
        "uncertainty method, Q90, and available countries. "
        "No filesystem paths are exposed."
    ),
)
async def get_forecast_metadata() -> ForecastMetadataResponse:
    """Return model and artifact provenance metadata."""
    try:
        metadata_raw = forecast_service.get_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast artifact unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forecast artifact is invalid or corrupted.",
        ) from exc
    return ForecastMetadataResponse(**metadata_raw)


# ---------------------------------------------------------------------------
# Phase 13 Step 1 — Production inference endpoints (errors sanitized in Step 2)
# ---------------------------------------------------------------------------

from app.schemas.forecast import InferenceRequest, InferenceResponse, ModelIdentityResponse


@router.get(
    "/model-info",
    response_model=ModelIdentityResponse,
    summary="Get Production Model Identity",
    description="Returns metadata and status for the frozen Phase 11 production model.",
)
async def get_model_info() -> ModelIdentityResponse:
    from app.ml.phase11_production_model import production_model, FrozenModelVerificationError
    try:
        meta = production_model.get_metadata()
        return ModelIdentityResponse(**meta)
    except FrozenModelVerificationError as exc:
        # Log full diagnostic server-side; NEVER forward hash/path details to caller
        logger.error(
            "model_info_verification_failed",
            event="FrozenModelVerificationError on /model-info",
            error_category="artifact_integrity_failure",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production model integrity verification failed. Service is unavailable.",
        )
    except Exception:
        logger.exception("model_info_unexpected_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model identity.",
        )


@router.post(
    "/gdp",
    response_model=InferenceResponse,
    summary="Predict GDP Growth using Frozen Production Model",
    description=(
        "Performs inference using the exactly 31 features required by the "
        "Phase 11 frozen model. Structured observability logging is applied "
        "to every request."
    ),
)
async def predict_gdp(request: InferenceRequest) -> InferenceResponse:
    from app.ml.phase11_production_model import production_model, FrozenModelVerificationError

    _t_start = time.monotonic()
    _request_ts = datetime.now(timezone.utc).isoformat()

    try:
        # Extract the 31 feature values (exclude routing fields)
        req_dict = request.model_dump()
        features = {k: v for k, v in req_dict.items() if k not in ("country", "year")}

        # Perform inference through the verified Phase 11 production path
        prediction = production_model.predict(features)
        meta = production_model.get_metadata()

        _duration_ms = round((time.monotonic() - _t_start) * 1000, 2)

        logger.info(
            "inference_success",
            endpoint="POST /api/v1/forecasts/gdp",
            model_version=meta.get("model_version", "phase11"),
            production_status=meta.get("status", "PHASE 11 PRODUCTION FROZEN"),
            country=request.country,
            year=request.year,
            success=True,
            duration_ms=_duration_ms,
            request_timestamp=_request_ts,
        )

        return InferenceResponse(
            country=request.country,
            year=request.year,
            predicted_gdp_growth=prediction,
            model_version=meta["model_version"],
            production_status=meta["status"],
        )

    except FrozenModelVerificationError:
        _duration_ms = round((time.monotonic() - _t_start) * 1000, 2)
        # Log full internal details server-side; NEVER expose to caller
        logger.error(
            "inference_artifact_failure",
            endpoint="POST /api/v1/forecasts/gdp",
            model_version="phase11",
            production_status="PHASE 11 PRODUCTION FROZEN",
            country=request.country,
            year=request.year,
            success=False,
            failure_category="artifact_integrity_failure",
            duration_ms=_duration_ms,
            request_timestamp=_request_ts,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production model integrity verification failed. Service is unavailable.",
        )

    except ValueError as exc:
        _duration_ms = round((time.monotonic() - _t_start) * 1000, 2)
        logger.warning(
            "inference_validation_failure",
            endpoint="POST /api/v1/forecasts/gdp",
            model_version="phase11",
            production_status="PHASE 11 PRODUCTION FROZEN",
            country=request.country,
            year=request.year,
            success=False,
            failure_category="validation_error",
            duration_ms=_duration_ms,
            request_timestamp=_request_ts,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception:
        _duration_ms = round((time.monotonic() - _t_start) * 1000, 2)
        logger.error(
            "inference_unexpected_failure",
            endpoint="POST /api/v1/forecasts/gdp",
            model_version="phase11",
            production_status="PHASE 11 PRODUCTION FROZEN",
            country=request.country,
            year=request.year,
            success=False,
            failure_category="internal_error",
            duration_ms=_duration_ms,
            request_timestamp=_request_ts,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Contact the system administrator.",
        )


# ---------------------------------------------------------------------------
# Phase 13 Step 2 — Operational safety endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Production Model Health Check",
    description=(
        "Returns an operational health summary for the frozen Phase 11 production model. "
        "Checks artifact integrity, schema validity, model configuration, and Candidate B isolation. "
        "Does not expose filesystem paths, raw hash values, or internal exception details."
    ),
)
async def get_production_health() -> dict:
    """
    Operational health summary.

    Returns:
        status: "healthy" | "degraded" | "unavailable"
        Plus per-check summary fields. No raw hashes or paths are exposed.
    """
    from app.services.production_health import run_full_health_check
    try:
        result = run_full_health_check()
        return result
    except Exception:
        logger.error("health_endpoint_unexpected_error", exc_info=True)
        # Fail-closed: return unavailable, no internal detail
        return {
            "status": "unavailable",
            "service": "econosphere-ai-production-inference",
            "model_status": "unavailable",
            "artifact_integrity": "failed",
            "schema_integrity": "failed",
            "configuration_integrity": "failed",
            "candidate_b_isolation": "unknown",
            "production_status": "PHASE 11 PRODUCTION FROZEN",
        }


@router.get(
    "/readiness",
    summary="Production Service Readiness Gate",
    description=(
        "Strict readiness check: returns ready=true only when ALL of the following pass: "
        "(1) frozen artifact integrity, (2) manifest integrity, (3) model configuration, "
        "(4) 31-feature schema, (5) Candidate B isolation, (6) model availability. "
        "Fail-closed: any single failure returns ready=false."
    ),
)
async def get_production_readiness() -> dict:
    """
    Production readiness gate.

    Returns ready=true only when every critical production safety check passes.
    Fail-closed on any exception — returns ready=false with safe structured data.
    """
    from app.services.production_health import run_readiness_check
    try:
        result = run_readiness_check()
        return result
    except Exception:
        logger.error("readiness_endpoint_unexpected_error", exc_info=True)
        return {
            "ready": False,
            "model_version": "phase11",
            "production_status": "PHASE 11 PRODUCTION FROZEN",
            "checks": {
                "artifact_integrity": False,
                "model_configuration": False,
                "feature_schema": False,
                "candidate_b_isolation": False,
                "model_available": False,
            },
        }


# ---------------------------------------------------------------------------
# Per-country forecast endpoint (unchanged from Step 1)
# ---------------------------------------------------------------------------

@router.get(
    "/{country_code}",
    response_model=GDPForecastResponse,
    summary="Get Official GDP T+1 Forecast for a Specific Country",
    description=(
        "Returns the 2026 GDP growth forecast for a specific country (ISO 3166-1 alpha-3 code). "
        "The point forecast is the model's single best estimate. "
        "The 90% prediction interval is calibrated from out-of-sample validation residuals "
        "and does not guarantee actual GDP will fall within the stated range. "
        "Returns 404 if the country is not in the forecast set."
    ),
    responses={
        404: {"description": "Country not in forecast set"},
        503: {"description": "Forecast artifact unavailable"},
    },
)
async def get_country_forecast(country_code: str) -> GDPForecastResponse:
    """Return the official GDP growth forecast for a single country."""
    normalized_code = country_code.upper().strip()

    # Basic validation — reject obviously non-ISO3 codes without touching service
    if not normalized_code.isalpha() or len(normalized_code) != 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid country code '{country_code}'. Expected ISO 3166-1 alpha-3 (3 letters).",
        )

    try:
        result = forecast_service.get_forecast(normalized_code)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast artifact unavailable. The ML pipeline output has not been generated.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forecast artifact is invalid or corrupted.",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No official forecast available for country code '{normalized_code}'. "
                f"The production forecast covers: CHN, GBR, IND, JPN, USA."
            ),
        )

    return GDPForecastResponse(**result)
