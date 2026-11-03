# xArm7 model provenance

- Upstream repository: <https://github.com/google-deepmind/mujoco_menagerie>
- Upstream directory: `ufactory_xarm7/`
- Pinned commit: `c96a32d28fb5da84da38c1da4d749e7a13212855`
- Source file: `xarm7.xml`
- Upstream license: BSD-3-Clause (see `LICENSE`)
- Upstream description and URDF derivation notes: `README.md`

The upstream directory contents are vendored without kinematic edits. The
adapter derives the end-effector transform from the `link_tcp` site and records
the MJCF file SHA-256 in the resolved robot configuration and trial manifest.
