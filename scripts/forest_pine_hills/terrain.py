"""Hilly, bounded-slope heightfield, using the shared triangulation and sampler.

The origin is the take-off datum: z=0, metres, Z up. All terrain queries use
the actual triangles, so tree/robot AGL agrees with the physics surface.
"""
import numpy as np

from scene import Terrain, TerrainZ


def triangle_slopes(z, grid):
    """Slope magnitude of both triangles in each shared-diagonal grid cell."""
    a, b, c, d = z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:]
    return np.stack((np.hypot(b - a, d - b), np.hypot(d - c, c - a))) / grid


class PineHillsTerrain(Terrain):
    """100 m plot, 25 cm mesh, rounded ridge/hollows and a curved drainage gully."""
    grid = .25
    area = 100.
    flat_r = 3.8
    blend_r = 11.

    def __init__(self, rng):
        # The shared mesh/z_at implementations are inherited; only elevation differs.
        self.phase = rng.uniform(-.25, .25, 3)
        self.xs = np.linspace(-50, 50, 401)
        x, y = np.meshgrid(self.xs, self.xs, indexing="ij")
        r = np.hypot(x, y)
        u = np.clip((r - self.flat_r) / (self.blend_r - self.flat_r), 0, 1)
        pad_blend = u * u * (3 - 2 * u)
        z = (self.raw(x, y) - self.raw(0., 0.)) * pad_blend
        # Bound the actual mesh triangles, including the pad transition; do not
        # infer a safe collider slope from a smoothed normal or analytic derivative.
        self.height_scale = min(1., np.tan(np.deg2rad(28.5)) / triangle_slopes(z, self.grid).max())
        self.Z = z * self.height_scale

    def raw(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        p = self.phase
        hill = 3.9 * np.sin((x + .30 * y) / 26 + p[0])
        hill += 2.5 * np.sin((y - .18 * x) / 22 + p[1])
        ridge = 4.2 * np.exp(-((x - .45 * y - 24) / 14) ** 2)
        hollow = -2.6 * np.exp(-((x + 21) / 11) ** 2 - ((y - 12) / 17) ** 2)
        # A gently meandering shallow gully on the eastern slope.
        gully_x = 16 + 3.2 * np.sin(y / 15 + p[2])
        gully = -1.65 * np.exp(-((x - gully_x) / 4.5) ** 2) * (.6 + .4 * np.cos(y / 62))
        hummocks = .10 * np.sin(x / 2.4 + p[0]) * np.cos(y / 2.9 + p[1])
        hummocks += .045 * np.sin((x + y) / 1.7) * np.cos((x - y) / 2.1)
        return hill + ridge + hollow + gully + hummocks


def terrain_description(terrain):
    grid = float(terrain.xs[1] - terrain.xs[0])
    slopes = np.rad2deg(np.arctan(triangle_slopes(terrain.Z, grid)))
    return dict(extents_xy_m=[[-50., -50.], [50., 50.]], grid_m=grid,
                min_height_m=float(terrain.Z.min()), max_height_m=float(terrain.Z.max()),
                relief_m=float(np.ptp(terrain.Z)), max_triangle_slope_deg=float(slopes.max()),
                p95_triangle_slope_deg=float(np.percentile(slopes, 95)),
                spawn_xyz_m=[0., 0., .07], flat_pad_radius_m=3.8,
                triangle_count=int(2 * (len(terrain.xs) - 1) ** 2),
                suggested_scan_xy_m=[[-22., -22.], [22., 22.]],
                suggested_scan_agl_m=[2.5, 4.],
                note="Z is relative to the take-off pad, not AGL. Use terrain-following for the scan; "
                     "avoid assuming a constant world-Z clears the surrounding hills.")


def normal_at(terrain, x, y, step=.25):
    """Local support-plane normal; height queries still use the exact mesh."""
    dx = (terrain.z_at(x + step, y) - terrain.z_at(x - step, y)) / (2 * step)
    dy = (terrain.z_at(x, y + step) - terrain.z_at(x, y - step)) / (2 * step)
    n = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def trail_distance(x, y):
    """Approximate perpendicular distance to a subtle winding root/litter track.

    The centerline is x=.55*y+.45*sin(y/4.5), crossing the take-off clearing.
    It is a worn ecological gap, not a separately flattened road surface.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.abs(x - .55 * y - .45 * np.sin(y / 4.5)) / np.sqrt(1 + (.55 + .1 * np.cos(y / 4.5)) ** 2)


def from_stand(stand):
    return TerrainZ(stand["terrain"])
