"""Giant Demon Serpent configuration: palette, proportions and level-of-detail presets."""
from dataclasses import dataclass, field


@dataclass
class LOD:
    name: str
    head_around: int        # vertices around the head cross-section (body/tail use /2, /4)
    head_spacing: float     # ring spacing on the head (world units)
    body_k: float           # body ring spacing = body_k * local radius
    body_max: float         # max body ring spacing
    jaw_around: int
    horn_around: int
    horn_rings: int
    spine_around: int
    spine_rings: int
    spine_every: int        # 1 = all spines, 2 = skip the small alternates
    eye_seg: tuple
    plate_rings: int
    small_teeth: bool
    tex_res: int


HERO = LOD("hero", head_around=96, head_spacing=0.04, body_k=0.24, body_max=0.32, jaw_around=48,
           horn_around=14, horn_rings=30, spine_around=8, spine_rings=8, spine_every=1,
           eye_seg=(32, 20), plate_rings=3, small_teeth=True, tex_res=1024)

GAMEPLAY = LOD("gameplay", head_around=48, head_spacing=0.075, body_k=0.36, body_max=0.75, jaw_around=24,
               horn_around=8, horn_rings=12, spine_around=6, spine_rings=4, spine_every=2,
               eye_seg=(16, 10), plate_rings=2, small_teeth=False, tex_res=1024)


@dataclass
class SerpentPalette:
    # linear-space colours
    dorsal: tuple = (0.014, 0.011, 0.013)        # deep charcoal
    flank: tuple = (0.085, 0.006, 0.011)         # dark crimson
    flank_low: tuple = (0.15, 0.010, 0.016)      # crimson
    secondary: tuple = (0.07, 0.008, 0.045)      # red-violet
    marking: tuple = (0.10, 0.012, 0.055)        # dorsal diamonds
    rim: tuple = (0.30, 0.06, 0.015)             # warm scale-edge highlight
    belly: tuple = (0.23, 0.065, 0.040)          # belly plates
    belly_edge: tuple = (0.50, 0.17, 0.055)
    horn_root: tuple = (0.10, 0.008, 0.012)
    horn_mid: tuple = (0.040, 0.030, 0.034)
    horn_tip: tuple = (0.16, 0.14, 0.14)
    bone_root: tuple = (0.30, 0.12, 0.07)
    bone_tip: tuple = (0.86, 0.78, 0.62)
    eye: tuple = (1.0, 0.42, 0.03)               # glowing amber
    throat: tuple = (1.0, 0.16, 0.02)            # inner fire
    mouth: tuple = (0.10, 0.008, 0.012)
    tongue: tuple = (0.26, 0.025, 0.04)
    iris_ring: tuple = (0.22, 0.02, 0.0)
    pupil: tuple = (0.004, 0.002, 0.002)
    glint: tuple = (0.95, 0.92, 0.85)
    body_metal: float = 0.0
    body_rough: float = 0.40
    horn_metal: float = 0.9


@dataclass
class SerpentConfig:
    name: str = "GiantDemonSerpent"
    head_len: float = 3.0
    head_scale: float = 1.24      # cross-section scale of the head (pet-style big head)
    eye_radius: float = 0.255
    jaw_open_deg: float = 13.0
    palette: SerpentPalette = field(default_factory=SerpentPalette)
    seed: int = 7
