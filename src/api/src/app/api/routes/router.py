"""Bundling of endpoint routers.

Import and add all endpoint routers here.
"""
from fastapi import APIRouter

from api.src.app.api.routes import training
from api.src.app.api.routes import inference
from api.src.app.core.tags_metadata import training_tag, inference_tag, general_tag


router = APIRouter()

router.include_router(training.router, prefix="/training", tags=[training_tag.name])
router.include_router(inference.router, prefix="/inference", tags=[inference_tag.name])

