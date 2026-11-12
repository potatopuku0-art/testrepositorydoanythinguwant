"""Quick preview: build base hero serpent and render a fast low-sample check image.
blender -b -P scripts/quick_preview.py -- out.png [variant] [lod] [view]"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import numpy as np  # noqa: E402
from pmcore import bl  # noqa: E402
from serpent.config import SerpentConfig, HERO, GAMEPLAY  # noqa: E402
from serpent.build import SerpentBuild, base_material_specs, realise  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = argv[0] if argv else "/tmp/q.png"
variant = argv[1] if len(argv) > 1 else "base"
lod = GAMEPLAY if (len(argv) > 2 and argv[2] == "gameplay") else HERO
view = argv[3] if len(argv) > 3 else "three_quarter"

t0 = time.time()
bl.reset_scene()
cfg = SerpentConfig()
b = SerpentBuild(cfg, lod)
specs, F = base_material_specs(os.path.join(ROOT, "output", "tmp", "tex"), cfg)
if variant == "hacker":
    from mutations.hacker import apply_hacker, HackerConfig
    specs = apply_hacker(b, specs, HackerConfig(), os.path.join(ROOT, "output", "tmp", "tex"), F)
print("build", time.time() - t0, "tris", b.tri_count())
for p in b.parts:
    print("  ", p.name, p.mesh.tri_count())
objs, _ = realise(b, specs, "GDS_" + variant)
print("realise", time.time() - t0)

import render_setup  # noqa: E402
render_setup.studio(variant, objs, b, view, out, samples=int(os.environ.get("SAMPLES", "24")),
                    res=(int(os.environ.get("RESX", "960")), int(os.environ.get("RESY", "540"))))
print("total", time.time() - t0)
