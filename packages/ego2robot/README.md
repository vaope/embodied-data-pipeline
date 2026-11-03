# Optional Ego2Robot package

This package contains the actual reproduction/engineering sources, model geometry, configuration profiles and worker entry points. It is not the authors' official implementation.

Install from the repository root with `python -m pip install -e ./packages/ego2robot`. Select video/data/robot extras for the stages you use. For now, use an editable source installation: the robot adapters and isolated workers resolve geometry and scripts from this checkout. A standalone wheel with all robot resources is not claimed.

From this package directory, `e2r --help` lists the available commands. Configure real data, base transforms and required runtimes explicitly. `m5`, `m5-inpaint` and `m6` profiles contain clearly marked runtime placeholders; install upstream models/weights separately and set those paths for your machine. No model repositories, weights, raw datasets or review approvals are bundled.

Read [system design](docs/system-design.md), [implementation order](docs/implementation-plan.md) and [working rules](AGENTS.md) before modifying stages. Included design/workplan documents contain historical milestone records; their old approvals do not transfer to new source/config fingerprints. Current geometry, temporal continuity and visual issues require their own review.

Source migration changes model-location examples and package metadata only; it does not adjust IK, retargeting, coordinate systems, geometry gates or visual algorithms. No new automated test cases or model/IK verification runs are introduced for this package. Geometry and temporal outputs retain the original image/video evidence and user-review requirements.

Robot model terms remain beside each model under assets/mjcf. The original-code project license is still pending selection; see ../../docs/THIRD_PARTY.md.
