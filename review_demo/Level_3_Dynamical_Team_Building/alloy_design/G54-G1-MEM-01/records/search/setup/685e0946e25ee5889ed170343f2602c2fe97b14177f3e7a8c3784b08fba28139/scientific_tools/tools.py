"""Small agent interfaces; scientific execution is isolated in a contained worker."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Annotated
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
KINDS = ("coarsening", "phase_stability", "solidification")


def candidate(value):
    """Validate explicit atomic percentages; never guess or normalize a basis."""
    if not isinstance(value, dict) or set(value) not in (
        {"Ni", "Er", "Zr"},
        {"Ni", "Er", "Zr", "Y", "Yb"},
    ):
        raise ValueError(
            "Specify Ni, Er and Zr in at.%; optional Y and Yb must both be zero; Al is balance"
        )
    result = {}
    for e in ("Ni", "Er", "Zr", "Y", "Yb"):
        v = value.get(e, 0.0)
        if (
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or v < 0
        ):
            raise ValueError(f"{e} must be a finite nonnegative atomic percentage")
        if e in ("Y", "Yb") and v != 0:
            raise ValueError(
                "Candidate Y and Yb must be zero in this alloy-design case"
            )
        result[e] = float(v)
    if sum(result.values()) >= 100:
        raise ValueError(
            "Al balance must be positive; do not supply normalized weight percentages"
        )
    return result


class AlloyDesignTools:
    """Host setup: choose an output folder inside the working repository.

    Register only the three evaluate_* bound methods as agent tools. They take
    compositions only; settings, reference inputs and storage are host-owned.
    """

    def __init__(self, output_directory):
        self.output_directory = Path(output_directory).resolve()
        if not self.output_directory.is_relative_to(ROOT):
            raise ValueError(
                "Keep tool outputs under experiments/management_alloy_design"
            )

    def _call(self, kind, compositions):
        settings = json.loads((HERE / "settings.json").read_text())
        if (
            not isinstance(compositions, list)
            or len(compositions) > settings["max_batch_size"]
        ):
            raise ValueError(
                f"Use a list of at most {settings['max_batch_size']} compositions"
            )
        comps = [candidate(c) for c in compositions]
        folder = self.output_directory / "calls" / uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        request = dict(kind=kind, compositions=comps)
        (folder / "request.json").write_text(
            json.dumps(request, indent=2, allow_nan=False) + "\n"
        )
        # The scientific child does not need access to project API credentials.
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(x in k.upper() for x in ("OPENAI", "API_KEY", "API_TOKEN"))
        }
        command = [sys.executable, "-B", str(HERE / "worker.py"), str(folder)]
        if env.get("ALLOYGEN_CONTAINED") != "1":
            command = [
                sys.executable,
                "-B",
                str(REPO / "experiments/tool_using_demo_reproduction/run_contained.py"),
                "--script",
                str(HERE / "worker.py"),
                str(folder),
            ]
        proc = subprocess.run(command, env=env, capture_output=True, text=True)
        log = proc.stdout + "\n" + proc.stderr
        for key in ("LSHOST", "LSFORCEHOST"):
            if env.get(key):
                log = log.replace(env[key], "[license host redacted]")
        (folder / "worker.log").write_text(log)
        path = folder / "result.json"
        if not path.exists():
            result = dict(
                status="failed",
                error="Scientific worker did not return a result; inspect worker.log",
                record=str(folder.relative_to(REPO)),
                worker_exit_code=proc.returncode,
            )
            path.write_text(json.dumps(result, indent=2) + "\n")
            return result
        return json.loads(path.read_text())

    def evaluate_coarsening(
        self,
        compositions: Annotated[
            list[dict[str, float]],
            "Ni, Er, Zr in at.%; Al balance; Y/Yb zero. Empty list returns the fixed benchmark.",
        ],
    ) -> dict:
        """Evaluate normalized L12 coarsening metric J at 250 C. Lower J is better; the fixed benchmark has J=1. Undefined phase states return null. Includes reference and saved evidence."""
        return self._call("coarsening", compositions)

    def evaluate_phase_stability(
        self,
        compositions: Annotated[
            list[dict[str, float]],
            "Ni, Er, Zr in at.%; Al balance; Y/Yb zero. Empty list returns the fixed benchmark.",
        ],
    ) -> dict:
        """Evaluate equilibrium L12 and Al3Zr mol% at 250/660 C and upper Al3Zr disappearance temperature. Report liquid involvement and lower disappearance intervals. Includes reference and saved evidence."""
        return self._call("phase_stability", compositions)

    def evaluate_solidification(
        self,
        compositions: Annotated[
            list[dict[str, float]],
            "Ni, Er, Zr in at.%; Al balance; Y/Yb zero. Empty list returns the fixed benchmark.",
        ],
    ) -> dict:
        """Evaluate Scheil terminal ternary/L12/Al3Ni/Al3Zr mol%, total-solid FR, interpolated CSC and HCS=CSC*FR/100. Lower cracking indicators are preferred. Includes reference and saved phase/heat curves."""
        return self._call("solidification", compositions)
