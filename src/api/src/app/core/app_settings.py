"""Settings that will be used for teh app."""
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from api.src.app.core.tags_metadata import metadata_tags


class AppSettings():

    # FastAPI App settings
    debug: bool = False
    docs_url: str = "/docs"
    openapi_prefix: str = ""
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/redoc"
    title: str = "Coin Type Classification Service"
    summary: str = "Make our computer vision service not suck as much🚀"
    version: str = "1.0"
    description: str = """
    ## API interface for our computer vision service🔥
    
    ---
    ---

    Through this interface we want to make our service accessible
    and easy to interact with for humans as well as other programs💯

    We provide endpoints to get general information about the trained models,
    the best models as well as allow for training new models easily by exposing
    the timm library with hundreds of pretrained models in the background. 
    We also allow for easy inference of new images💦
    """

    
    @property
    def fastapi_kwargs(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "docs_url": self.docs_url,
            "openapi_prefix": self.openapi_prefix,
            "openapi_url": self.openapi_url,
            "redoc_url": self.redoc_url,
            "title": self.title,
            "version": self.version,
            "summary": self.summary,
            "description": self.description,
        }
