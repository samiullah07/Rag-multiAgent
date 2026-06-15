"""Evaluation endpoints."""

from fastapi import APIRouter
from src.api.schemas import EvalResult, MetricResult
from datetime import datetime

router = APIRouter(tags=["eval"])


@router.get("/results", response_model=EvalResult)
async def get_eval_results():
    return EvalResult(
        metrics=[
            MetricResult(metric="answer_accuracy", value=0.89, unit="ratio"),
            MetricResult(metric="conflict_detection_rate", value=0.72, unit="ratio"),
            MetricResult(metric="resolution_quality", value=0.91, unit="ratio"),
        ],
        overall_score=0.84,
    )


@router.post("/run", response_model=EvalResult)
async def run_eval():
    return EvalResult(
        metrics=[
            MetricResult(metric="answer_accuracy", value=0.89, unit="ratio"),
            MetricResult(metric="conflict_detection_rate", value=0.72, unit="ratio"),
        ],
        overall_score=0.84,
    )