from fastapi import APIRouter, Query
from typing import List
import numpy as np
import logging

router = APIRouter(prefix="/analysis", tags=["Analysis Agent"])
logger = logging.getLogger("finbreaker")

@router.post("/risk_exposure")
def risk_exposure(
    allocations: List[float],
    region: List[str],
    sector: List[str],
    target_region: str = Query("Asia"),
    target_sector: str = Query("Tech")
):
    logger.info(f"Calculating risk exposure for region={target_region}, sector={target_sector}")
    exposure = sum([
        alloc for alloc, reg, sec in zip(allocations, region, sector)
        if target_region.lower() in reg.lower() and target_sector.lower() in sec.lower()
    ])
    logger.info(f"Exposure calculated: {exposure}")
    return {"exposure": exposure}

@router.get("/")
def root():
    return {"status": "Analysis Agent running"}
