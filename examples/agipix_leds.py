"""Opt-in navigation lights, attached once to the rigid body (metres, FLU).

Motor axes and ring radius/height come from setting_H_geometry.json. Each bulb
is tangent to the outer rim on the body-centre-to-motor diagonal. All geometry
is visual only: no mass, collision, sensor, publisher or update callback.
"""

import math


MOTOR_X, MOTOR_Y = 0.09856, 0.13150
RING_RADIUS = 0.0873
BULB_RADIUS = 0.004
HEIGHT = 0.030
EMISSION = 8.0
LIGHT_INTENSITY = 40.0
LIGHT_RADIUS = 0.004
CONE_DEGREES = 75.0


def mounts():
    """Yield name, centre, outward direction and RGB colour in body coordinates."""
    length = math.hypot(MOTOR_X, MOTOR_Y)
    for name, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        direction = (sx * MOTOR_X / length, sy * MOTOR_Y / length, 0.0)
        radius = RING_RADIUS + BULB_RADIUS
        position = (sx * MOTOR_X + radius * direction[0],
                    sy * MOTOR_Y + radius * direction[1], HEIGHT)
        yield name, position, direction, (1.0, 1.0, 1.0) if sx > 0 else (1.0, 0.0, 0.0)


def add_navigation_lights(stage, body_path):
    """Author four outward lights. Called only when --leds is enabled."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    if not stage.GetPrimAtPath(body_path):
        raise ValueError(f"Missing drone body: {body_path}")
    root = body_path + "/NavigationLights"
    UsdGeom.Xform.Define(stage, root)
    for name, position, direction, color in mounts():
        path = root + "/" + name
        mount = UsdGeom.Xform.Define(stage, path)
        mount.AddTranslateOp().Set(Gf.Vec3d(*position))
        # USD lights emit along local -Z. No transform reset: inherit body pose.
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*direction))
        mount.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
        material = UsdShade.Material.Define(stage, path + "/Material")
        shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * EMISSION)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        bulb = UsdGeom.Sphere.Define(stage, path + "/Bulb")
        bulb.CreateRadiusAttr(BULB_RADIUS)
        UsdShade.MaterialBindingAPI.Apply(bulb.GetPrim()).Bind(material)
        light = UsdLux.SphereLight.Define(stage, path + "/Light")
        # Place the analytic emitter just beyond the opaque visual shell; the
        # nearest part of its sphere touches the bulb's outward face.
        light.AddTranslateOp().Set(Gf.Vec3d(0, 0, -2 * BULB_RADIUS))
        light.CreateRadiusAttr(LIGHT_RADIUS)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateIntensityAttr(LIGHT_INTENSITY)
        light.CreateExposureAttr(0.0)
        # Modest fixed power; inverse-square falloff and an outward cone keep
        # useful illumination local. No unsupported hard-distance cutoff attr.
        light.CreateNormalizeAttr(True)
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(CONE_DEGREES)
        shaping.CreateShapingConeSoftnessAttr(0.3)
        light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    return root
