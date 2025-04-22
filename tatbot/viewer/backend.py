from fastapi import FastAPI, Body
from pxr import Usd, UsdGeom, Gf
from pathlib import Path

STAGE_PATH = Path("/tatbot/output/current_scene.usd")
app = FastAPI()

@app.post("/save-xforms")
async def save_xforms(xforms: dict = Body(...)):
    stage = Usd.Stage.Open(str(STAGE_PATH))
    for prim_path, mat in xforms.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
        op.Set(Gf.Matrix4d(*mat))
    stage.GetRootLayer().Save()
    return {"status": "ok"}