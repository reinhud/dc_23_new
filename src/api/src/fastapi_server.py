from fastapi import FastAPI

from api.src.app.api.routes.router import router
from api.src.app.core.config import add_middleware, get_app_settings
from pytorch_service.src.sample_main import foo
from pytorch_service.src.data.datasets.coin_data import CoinData
from pytorch_service.src.training.config.train_config import TrainConfig
from pytorch_service.src.training.training import Training


def get_app() -> FastAPI:
    """Instanciating and setting up FastAPI application."""
    settings = get_app_settings()

    app = FastAPI(**settings.fastapi_kwargs)

    add_middleware(app)

    app.include_router(router)

    @app.on_event("startup")
    async def startup_event() -> None:
        pass

    return app


app = get_app()



# ===== App Info Endpoints ===== #
@app.get("/")
async def root():

    return {"message": "OK Eric"}

@app.get("/cwd")
async def get_cwd():
    import os

    return {"message": {os.getcwd()}}

@app.get("/pythonpath")
async def rooget_pythonpathst():

    import sys

    print(sys.path)

    return {"message": {os.getcwd()}}








