import os, torch, uvicorn, random
import numpy as np
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager

from src.multitex_cvae.data import build_condition_vector, condition_vocab_from_dict
from src.multitex_cvae.model import MultiTexCVAE

# --- СЛОВАРИ ДАННЫХ ---
PLANET_PALETTES = {'tropical': 'green jungle colors', 'volcanic': 'lava orange and black', 'crystal': 'purple blue crystal', 'toxic': 'neon green, sickly yellow', 'frozen': 'ice blue and white', 'desert': 'sand beige', 'nebular': 'space purple and gold', 'fungal': 'moss green and violet', 'aquatic': 'aqua blue ocean', 'metallic': 'steel silver', 'lush': 'fresh leaf green', 'artic': 'pale ice blue', 'synthetic': 'neon magenta and cyan'}
PLANET_MICROSTRUCTURES = {'tropical': 'fibrous plant strands', 'volcanic': 'porous lava rock', 'crystal': 'angular crystal facets', 'toxic': 'slimy gel bubbles', 'frozen': 'compacted ice grains', 'desert': 'fine sand grains', 'nebular': 'soft cloudy particles', 'fungal': 'porous sponge tissue', 'aquatic': 'smooth wet film', 'metallic': 'brushed micro grooves', 'lush': 'spongy plant tissue', 'artic': 'coarse snow grains', 'synthetic': 'geometric foam cells'}
PLANET_TOPOLOGIES = {'tropical': 'dense foliage', 'volcanic': 'cracked basalt', 'crystal': 'hex grid', 'toxic': 'bubble cells', 'frozen': 'horizontal layers', 'desert': 'sand ripples', 'nebular': 'swirling filaments', 'fungal': 'radial gills', 'aquatic': 'water ripples', 'metallic': 'panel seams', 'lush': 'dense leaf canopy', 'artic': 'ice polygon cracks', 'synthetic': 'tech hex grid'}
PLANET_FINISHES = {'tropical': 'wet glossy', 'volcanic': 'rough matte', 'crystal': 'clear glossy', 'toxic': 'wet slimy', 'frozen': 'frosted matte', 'desert': 'dry matte', 'nebular': 'glowing ethereal', 'fungal': 'moist velvety', 'aquatic': 'water glossy', 'metallic': 'brushed metal', 'lush': 'leaf glossy', 'artic': 'snow matte', 'synthetic': 'plastic glossy'}
BIOME_CONTEXT = {'tropical': 'tropical ground', 'volcanic': 'volcanic terrain', 'crystal': 'crystal surface', 'toxic': 'toxic surface', 'frozen': 'frozen landscape', 'desert': 'desert surface', 'nebular': 'nebular surface', 'fungal': 'fungal surface', 'aquatic': 'ocean floor', 'metallic': 'metallic ground', 'lush': 'lush surface', 'artic': 'arctic surface', 'synthetic': 'synthetic surface'}

STATE = {}
STATIC_DIR = "static_textures"

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_path = os.environ.get("CVAE_CHECKPOINT", "checkpoints/multitex_cvae_mps_long.pth")
    if not os.path.exists(ckpt_path):
        print(f"!!! ФАЙЛ МОДЕЛИ НЕ НАЙДЕН: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = condition_vocab_from_dict(ckpt["vocab"])
    cfg = ckpt.get("config", {})
    model = MultiTexCVAE(cond_dim=vocab.total_dim, z_dim=int(cfg.get("z_dim", 64)), 
                         base_channels=int(cfg.get("base_channels", 64)), image_size=int(cfg.get("img_size", 64)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    STATE.update({
        "model": model, "vocab": vocab, "z_dim": int(cfg.get("z_dim", 64)), 
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })
    model.to(STATE["device"])
    print(f"🚀 CVAE готова на {STATE['device']}")
    yield
    STATE.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs(STATIC_DIR, exist_ok=True)

async def perform_generation(biome: str, seed: int, type: str):
    # Фоллбэк значения
    palette = PLANET_PALETTES.get(biome, "brown earthy colors")
    micro = PLANET_MICROSTRUCTURES.get(biome, "rough grains")
    topo = PLANET_TOPOLOGIES.get(biome, "uneven surface")
    finish = PLANET_FINISHES.get(biome, "matte")
    general = BIOME_CONTEXT.get(biome, "terrain")

    # Специальная настройка для DIRT
    if type == "dirt":
        palette = "very dark chocolate brown soil, deep sienna, earthy umber"
        ctx = "dense underground dirt layer, organic soil, no light"
        micro = "compacted dirt particles"
        finish = "rough matte"
    elif type == "ground":
        ctx = f"top surface ground of {biome}"
    elif type == "bark":
        ctx = f"tree trunk bark of {biome}"
    elif type == "leaves":
        ctx = f"foliage and leaves of {biome}"
    else:
        ctx = general

    meta = {"biome": biome, "color": palette, "micro": micro, "topo": topo, "context": ctx, "finish": finish}

    torch.manual_seed(seed)
    z = torch.randn(1, STATE["z_dim"]).to(STATE["device"])
    cond = build_condition_vector(meta, STATE["vocab"]).unsqueeze(0).to(STATE["device"])

    with torch.no_grad():
        maps = STATE["model"].decode(z, cond)

    biome_path = os.path.join(STATIC_DIR, biome)
    os.makedirs(biome_path, exist_ok=True)

    for map_name, map_tensor in maps.items():
        # Регулировка яркости: dirt делаем самым темным
        t = map_tensor[0].detach().cpu().clamp(0, 1)
        
        if type == "dirt":
            t = t * 0.3  # Глубокий темный коричневый
        elif type == "bark":
            t = t * 0.45 # Стандартное затемнение
        
        img_np = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        
        # Если тензор одноканальный (ч/б), переводим в RGB
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        save_path = os.path.join(biome_path, f"{type}_{map_name}.png")
        Image.fromarray(img_np).save(save_path)
    
    return True

@app.exception_handler(404)
async def static_404_handler(request, exc):
    path = request.url.path
    if path.startswith("/static/"):
        parts = path.split("/") # ['', 'static', 'biome', 'filename']
        if len(parts) >= 4:
            biome = parts[2]
            filename = parts[3]
            tex_type = filename.split("_")[0] 
            
            print(f"🛠 Генерирую текстуру: {biome}/{filename}")
            await perform_generation(biome, random.randint(1, 99999), tex_type)
            
            full_path = os.path.join(STATIC_DIR, biome, filename)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    return Response(content=f.read(), media_type="image/png")
    
    return JSONResponse(status_code=404, content={"error": "Not Found"})

# Монтируем статику после хендлера 404
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/generate/{biome}")
async def api_generate(biome: str, seed: int = 42, type: str = "ground"):
    await perform_generation(biome, seed, type)
    return {"status": "ok", "path": f"/static/{biome}/{type}_albedo.png"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)