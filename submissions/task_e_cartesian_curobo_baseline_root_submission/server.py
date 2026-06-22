from fastapi import FastAPI, File, Form, Request, UploadFile
from typing import Optional
import logging
import os
import signal
import sys
import threading

import numpy as np
import torch


def setup_logging():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.environ.get("LOG_DIR", os.path.join(project_dir, "logs"))
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, "user.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if os.environ.get("LOG_TO_CONSOLE"):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(handler.formatter)
        logger.addHandler(console)
    return logger


logger = setup_logging()

try:
    from solution import AlgSolution

    agent = AlgSolution()
except Exception:
    import traceback

    logger.error("Failed to initialize AlgSolution: %s", traceback.format_exc())
    raise


app = FastAPI()
logger.info("Server started")


def tensor_from_upload(data: bytes, dtype, shape):
    tensor = torch.tensor(np.frombuffer(data, dtype=dtype).reshape(*shape))
    return tensor.cuda() if torch.cuda.is_available() else tensor


@app.post("/step")
async def step(
    proprio: UploadFile = File(),
    extero: Optional[UploadFile] = File(None),
    head_rgb: Optional[UploadFile] = File(None),
    head_depth: Optional[UploadFile] = File(None),
    ee_rgb: Optional[UploadFile] = File(None),
    ee_depth: Optional[UploadFile] = File(None),
    video_rgb: Optional[UploadFile] = File(None),
    video_depth: Optional[UploadFile] = File(None),
    current_score: float = Form(),
):
    proprio_t = tensor_from_upload(await proprio.read(), np.float32, (1, -1))
    extero_t = (
        tensor_from_upload(await extero.read(), np.float32, (1, -1))
        if extero is not None
        else None
    )

    image = {}
    if head_rgb is not None:
        image["head_rgb"] = tensor_from_upload(await head_rgb.read(), np.uint8, (1, 480, 640, 3))
    if head_depth is not None:
        image["head_depth"] = tensor_from_upload(await head_depth.read(), np.float32, (1, 480, 640, 1))
    if ee_rgb is not None:
        image["ee_rgb"] = tensor_from_upload(await ee_rgb.read(), np.uint8, (1, 480, 640, 3))
    if ee_depth is not None:
        image["ee_depth"] = tensor_from_upload(await ee_depth.read(), np.float32, (1, 480, 640, 1))
    if video_rgb is not None:
        image["video_rgb"] = tensor_from_upload(await video_rgb.read(), np.uint8, (1, 480, 640, 3))
    if video_depth is not None:
        image["video_depth"] = tensor_from_upload(await video_depth.read(), np.float32, (1, 480, 640, 1))

    return agent.predicts(
        obs={"proprio": proprio_t, "extero": extero_t, "image": image},
        current_score=current_score,
    )


@app.post("/reset")
async def reset(request: Request):
    form_data = await request.json()
    agent.reset(**form_data)
    return {"message": "success"}


@app.get("/synchronize")
async def synchronize():
    return {"message": "success"}


@app.get("/health")
async def health():
    return {"message": "success"}


@app.post("/stop")
async def stop(request: Request):
    body = await request.json()
    logger.info("Stop message received: %s", body.get("msg"))
    return {"message": "success"}


@app.post("/quit")
async def quit(request: Request):
    body = await request.json()
    logger.info("Quit message received: %s", body.get("msg", "quit"))

    def shutdown_server():
        logger.info("Shutting down the server...")
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Timer(1.0, shutdown_server).start()
    return {"message": "Server is shutting down gracefully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
