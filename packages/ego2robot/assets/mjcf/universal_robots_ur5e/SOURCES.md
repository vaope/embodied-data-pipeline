# UR5e model provenance

- Upstream repository: <https://github.com/google-deepmind/mujoco_menagerie>
- Pinned revision: `c96a32d28fb5da84da38c1da4d749e7a13212855`
- Source directory: `universal_robots_ur5e/`
- Source MJCF: `universal_robots_ur5e/ur5e.xml`
- Upstream license: BSD-3-Clause; the original `LICENSE` and `README.md` are retained.
- Source files are kept unmodified. `scene.xml`, `CHANGELOG.md`, `ur5e.png`, and the complete `assets/` mesh directory are included from the same revision.
- Menagerie model definition is an arm-only UR5e. It contains six revolute arm joints and the official `attachment_site`; it does not contain a gripper, jaw geometry, gripper joints, or a jaw-width command mapping.
- The adapter uses the source `home` keyframe and reads joint ranges from this MJCF. The manufacturer technical specification lists 850 mm reach; it also reports an elbow range of +/-160 degrees, while the pinned MJCF permits +/-3.1415 rad. That joint-limit discrepancy is surfaced in the adapter notes for a project ADR; the adapter currently reports the MJCF range and does not silently modify the upstream model.
- Manufacturer technical specification: <https://www.universal-robots.com/manuals/EN/TechSheets/UR5e_techsheet_pdf_online/UR5e_techsheet_en.pdf> (reach 850 mm; six rotating joints; elbow range discrepancy above).

The image in `ur5e.png` is the upstream Menagerie reference image. Verification render and FK/IK evidence are stored separately under `data/morphology_trials/ur5e/model-preview/`.
