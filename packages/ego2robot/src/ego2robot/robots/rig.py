"""Two-arm rig: one MuJoCo scene for cross-arm collision and joint geometry.

ADR-0022 decision 5: the first two-arm delivery uses **two explicit fixed bases**
and a joint scene, and only then looks at an automatic two-base search.  The IK
itself is *not* re-implemented here: each arm keeps the single-arm adapter
(``robots/ik.solve_trajectory``) so the bimanual path cannot silently change the
solver behaviour the single-arm artifacts were reviewed against.

What the joint scene is for
---------------------------
* cross-arm contacts (the paper's ``l1_cross_arm_contacts_max`` had no input until
  now);
* a self-check that the assembly is right: FK of each arm in the joint scene must
  reproduce the independent per-arm FK after the relative base is applied;
* per-arm link positions in the joint frame, used for the visibility proxy.

Assembly rules (ADR-0022 decision 3): every body and joint is renamed with an arm
prefix, and every reference (tendon, equality, actuator, contact exclude) is
rewritten; renaming only the root body would produce a scene that loads and then
behaves subtly wrong.

The generated XML is written next to the vendored model (where the project already
keeps ``.e2r_scene_*.xml``) so ``meshdir="assets"`` keeps resolving, and its name
carries a hash of the inputs so a changed placement cannot silently reuse a stale
scene.
"""

from __future__ import annotations

import copy
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.transforms import (
    invert_transform,
    make_transform,
    quat_xyzw_from_matrix,
)

__all__ = [
    "ARM_PREFIX",
    "DualRigScene",
    "build_dual_scene",
    "prefixed",
    "relative_placement",
]

#: Arm letters used in the generated model, so a body name alone says whose arm it is.
ARM_PREFIX = {"right": "R", "left": "L"}


def relative_placement(
    T_C_B_right: NDArray[np.float64], T_C_B_left: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``T_BR_BL``: left base expressed in the right base frame.

    The MuJoCo world of the joint scene is the **right** arm's base frame, which is
    the frame the existing single-arm camera/render code already uses.
    """
    return invert_transform(np.asarray(T_C_B_right, dtype=np.float64)) @ np.asarray(
        T_C_B_left, dtype=np.float64
    )


def prefixed(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def _quat_wxyz(R: NDArray[np.float64]) -> list[float]:
    x, y, z, w = quat_xyzw_from_matrix(R)
    return [float(w), float(x), float(y), float(z)]


def _strip_visual_geoms(body: ET.Element) -> None:
    for geom in list(body.findall("geom")):
        if geom.get("class") == "visual":
            body.remove(geom)
    for child in body.findall("body"):
        _strip_visual_geoms(child)


def _prefix_names(element: ET.Element, prefix: str) -> None:
    """Rename body/joint elements; leave mesh/material/class references alone."""
    if element.tag in {"body", "joint"}:
        name = element.get("name")
        if name:
            element.set("name", prefixed(prefix, name))
    for child in list(element):
        _prefix_names(child, prefix)


def _mesh_effective_name(mesh: ET.Element) -> str:
    name = mesh.get("name")
    if name:
        return name
    file = mesh.get("file") or ""
    return Path(file).stem


def _referenced_meshes(root: ET.Element) -> set[str]:
    return {geom.get("mesh") for geom in root.iter("geom") if geom.get("mesh")}


def _scene_hash(payload: dict[str, Any], source_sha: str) -> str:
    blob = repr(sorted(payload.items())).encode("utf-8") + source_sha.encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def build_dual_scene(
    source_xml: str | Path,
    relative_T_W_B_left: NDArray[np.float64],
    *,
    collision_only: bool = True,
    with_camera: bool = False,
    out_dir: str | Path | None = None,
) -> Path:
    """Write a deterministic two-Panda scene and return its path.

    ``relative_T_W_B_left`` is the left base pose in the right base frame, which is
    also the generated scene's MuJoCo world (the right arm sits at the identity).
    """
    source = Path(source_xml)
    if not source.is_file():
        raise FileNotFoundError(f"base MJCF not found: {source}")
    tree = ET.parse(source)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{source}: no <worldbody>")
    arm_bodies = [b for b in worldbody.findall("body") if b.get("name") == "link0"]
    if len(arm_bodies) != 1:
        raise ValueError(f"{source}: expected exactly one top-level <body name='link0'>")
    template = arm_bodies[0]

    T_W_BL = np.asarray(relative_T_W_B_left, dtype=np.float64)
    if T_W_BL.shape != (4, 4):
        raise ValueError(f"left-base placement must be 4x4, got {T_W_BL.shape}")
    if not np.allclose(T_W_BL[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("left-base placement must have a [0,0,0,1] bottom row")
    if not np.allclose(T_W_BL[:3, :3].T @ T_W_BL[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("left-base placement rotation is not orthonormal")
    if float(np.linalg.det(T_W_BL[:3, :3])) < 0.0:
        raise ValueError(
            "left-base placement has determinant -1: a reflection is not a rigid "
            "placement (ADR-0001 / ADR-0022 decision 2)"
        )
    placements = {
        "right": (np.zeros(3), [1.0, 0.0, 0.0, 0.0]),
        "left": (T_W_BL[:3, 3], _quat_wxyz(T_W_BL[:3, :3])),
    }

    # --- duplicate the arm -------------------------------------------------
    worldbody.remove(template)
    for side, prefix in ARM_PREFIX.items():
        body = copy.deepcopy(template)
        _prefix_names(body, prefix)
        pos, quat = placements[side]
        body.set("pos", " ".join(f"{v:.9g}" for v in np.asarray(pos, dtype=np.float64)))
        body.set("quat", " ".join(f"{v:.9g}" for v in quat))
        if collision_only:
            _strip_visual_geoms(body)
        worldbody.append(body)

    # --- drop meshes nothing references (the visual ones, in collision mode) --
    asset = root.find("asset")
    if collision_only and asset is not None:
        keep = _referenced_meshes(root)
        for mesh in list(asset.findall("mesh")):
            if _mesh_effective_name(mesh) not in keep:
                asset.remove(mesh)

    # --- per-arm tendon / equality / contact / actuator ---------------------
    for tag in ("contact", "tendon", "equality", "actuator"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)

    # MJCF element order matters: contact < tendon < equality < actuator
    contact = ET.SubElement(root, "contact")
    tendon = ET.SubElement(root, "tendon")
    equality = ET.SubElement(root, "equality")
    actuator = ET.SubElement(root, "actuator")
    for _side, prefix in ARM_PREFIX.items():
        ET.SubElement(
            contact, "exclude", body1=prefixed(prefix, "link0"), body2=prefixed(prefix, "link1")
        )
        fixed = ET.SubElement(tendon, "fixed", name=prefixed(prefix, "split"))
        for joint in ("finger_joint1", "finger_joint2"):
            ET.SubElement(fixed, "joint", joint=prefixed(prefix, joint), coef="0.5")
        ET.SubElement(
            equality,
            "joint",
            joint1=prefixed(prefix, "finger_joint1"),
            joint2=prefixed(prefix, "finger_joint2"),
            solimp="0.95 0.99 0.001",
            solref="0.005 1",
        )
        # Actuators mirror the source gains; they are not used by IK, but keeping
        # the wiring identical per arm makes the scene loadable and comparable.
        gains = {
            1: ("4500", "0 -4500 -450", None, None),
            2: ("4500", "0 -4500 -450", "-1.7628 1.7628", None),
            3: ("3500", "0 -3500 -350", None, None),
            4: ("3500", "0 -3500 -350", "-3.0718 -0.0698", None),
            5: ("2000", "0 -2000 -200", None, "-12 12"),
            6: ("2000", "0 -2000 -200", "-0.0175 3.7525", "-12 12"),
            7: ("2000", "0 -2000 -200", None, "-12 12"),
        }
        for i, (gain, bias, ctrlrange, forcerange) in gains.items():
            attrs: dict[str, str] = {
                "class": "panda",
                "name": prefixed(prefix, f"actuator{i}"),
                "joint": prefixed(prefix, f"joint{i}"),
                "gainprm": gain,
                "biasprm": bias,
            }
            if ctrlrange is not None:
                attrs["ctrlrange"] = ctrlrange
            if forcerange is not None:
                attrs["forcerange"] = forcerange
            ET.SubElement(actuator, "general", **attrs)
        ET.SubElement(
            actuator,
            "general",
            **{
                "class": "panda",
                "name": prefixed(prefix, "actuator8"),
                "tendon": prefixed(prefix, "split"),
                "forcerange": "-100 100",
                "ctrlrange": "0 255",
                "gainprm": "0.01568627451 0 0",
                "biasprm": "0 -100 -10",
            },
        )

    # keyframes cannot be prefixed and are not needed for geometry QC
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)

    if with_camera:
        # A world-level camera in the *right base* world; RigRenderer overwrites
        # cam_pos/cam_quat from T_C_B at runtime, so the placeholder pose is inert.
        ET.SubElement(
            worldbody,
            "camera",
            name="rig_cam",
            pos="1 0 1",
            quat="1 0 0 0",
            fovy="45",
        )

    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "left_pos": [float(v) for v in T_W_BL[:3, 3]],
        "left_quat": [float(v) for v in _quat_wxyz(T_W_BL[:3, :3])],
        "collision_only": bool(collision_only),
        "with_camera": bool(with_camera),
        "arms": sorted(ARM_PREFIX.items()),
    }
    digest = _scene_hash(payload, source_sha)
    target_dir = Path(out_dir) if out_dir is not None else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f".e2r_rig_{digest}.xml"
    tree.write(target, encoding="utf-8", xml_declaration=False)
    return target


@dataclass
class DualRigScene:
    """Two arms in one MuJoCo model: cross-arm contacts and joint-frame FK."""

    model_path: Path
    #: un-prefixed body names per side, so a prefixed name can be attributed
    arm_bodies: dict[str, Sequence[str]]
    arm_joint_names: dict[str, Sequence[str]]
    arm_gripper_joint_names: dict[str, Sequence[str]]
    arm_eef_body: dict[str, str]
    _scene: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._bodies: dict[str, set[str]] = {
            side: set(names) for side, names in self.arm_bodies.items()
        }
        for side, prefix in ARM_PREFIX.items():
            for name in list(self.arm_joint_names[side]) + list(self.arm_gripper_joint_names[side]):
                self._joint_id(prefixed(prefix, name))

    # -- naming ------------------------------------------------------------

    def _joint_id(self, joint_name: str) -> int:
        jid = int(self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if jid < 0:
            raise KeyError(f"{self.model_path}: joint {joint_name!r} missing from the joint scene")
        return jid

    def body_arm(self, body_name: str) -> str | None:
        """Which arm a body belongs to, from the generated name prefix."""
        for side, prefix in ARM_PREFIX.items():
            if body_name.startswith(prefix) and body_name[len(prefix) :] in self._bodies[side]:
                return side
        return None

    # -- state -------------------------------------------------------------

    def set_arm(
        self,
        side: str,
        qpos: NDArray[np.float64],
        gripper_qpos: NDArray[np.float64] | None = None,
    ) -> None:
        prefix = ARM_PREFIX[side]
        qpos = np.asarray(qpos, dtype=np.float64)
        for i, name in enumerate(self.arm_joint_names[side]):
            jid = self._joint_id(prefixed(prefix, name))
            self.data.qpos[int(self.model.jnt_qposadr[jid])] = qpos[i]
        if gripper_qpos is not None:
            gripper_qpos = np.asarray(gripper_qpos, dtype=np.float64)
            for i, name in enumerate(self.arm_gripper_joint_names[side]):
                if i >= gripper_qpos.size:
                    break
                jid = self._joint_id(prefixed(prefix, name))
                self.data.qpos[int(self.model.jnt_qposadr[jid])] = gripper_qpos[i]

    def forward(self) -> None:
        self._mujoco.mj_forward(self.model, self.data)

    def body_pose(self, body_name: str) -> NDArray[np.float64]:
        bid = int(self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name))
        if bid < 0:
            raise KeyError(f"{self.model_path}: body {body_name!r} missing")
        return make_transform(self.data.xmat[bid].reshape(3, 3), self.data.xpos[bid])

    def arm_tcp(self, side: str, T_E_TCP: NDArray[np.float64]) -> NDArray[np.float64]:
        prefix = ARM_PREFIX[side]
        return self.body_pose(prefixed(prefix, self.arm_eef_body[side])) @ np.asarray(
            T_E_TCP, dtype=np.float64
        )

    def cross_arm_contacts(self) -> tuple[int, list[tuple[str, str]]]:
        """``(contact_count, unique_body_pairs)`` for contacts between the two arms.

        Both numbers are reported: the threshold is defined on *frames* with more
        than N contacts, while the deduplicated pair list says which bodies touch.
        Mixing the two units has already produced misleading numbers in this
        project (PIT-047), so they are kept separate.
        """
        count = 0
        pairs: set[tuple[str, str]] = set()
        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            b1 = self._mujoco.mj_id2name(
                self.model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[contact.geom1]),
            )
            b2 = self._mujoco.mj_id2name(
                self.model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[contact.geom2]),
            )
            arm1 = self.body_arm(b1 or "")
            arm2 = self.body_arm(b2 or "")
            if arm1 is None or arm2 is None or arm1 == arm2:
                continue
            count += 1
            pairs.add(tuple(sorted((b1 or "?", b2 or "?"))))
        return count, sorted(pairs)
