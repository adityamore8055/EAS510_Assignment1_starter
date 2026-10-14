"""EAS 510 - Project 1: SimpleDetector (the rule-based expert system).

A detective registers a folder of known originals, then evaluates any suspect
image: every registered rule scores the suspect against each original, the
scores are weighted into a 0-100 confidence, and the highest-confidence
original is the match (or REJECTED if below the threshold).
"""

import os
from pathlib import Path

import rules


MATCH_THRESHOLD = 50.0  # confidence below this -> REJECT


class SimpleDetector:
    def __init__(self, rule_names=None, rule_modules=None):
        self.targets = {}  # original stem -> {"stem", "path"}
        #: V1 rules are Rules 1-3. Phase 2 appends rule4_* via rule_names.
        self.rule_names = list(rule_names) if rule_names else list(rules.RULES)
        self.rule_modules = list(rule_modules) if rule_modules else [rules]

    def register_targets(self, folder):
        """Load (by path) every image in `folder` as a known original."""
        folder = str(folder)
        matches = sorted(p for p in self._image_paths(folder))
        if not matches:
            raise FileNotFoundError(f"no images found under {folder!r}")
        self.targets = {
            stem: {"stem": stem, "path": path}
            for stem, path in matches
        }
        return self.targets

    @staticmethod
    def _image_paths(folder):
        exts = {".jpg", ".jpeg", ".png"}
        for entry in sorted(os.listdir(folder)):
            path = os.path.join(folder, entry)
            if os.path.isfile(path) and Path(entry).suffix.lower() in exts:
                yield os.path.splitext(entry)[0], path

    def _resolve_rule(self, name):
        """Callable, or a function found in one of the rule modules."""
        if callable(name):
            return name
        for mod in self.rule_modules:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(f"no rule function named {name!r}")

    def evaluate(self, input_path):
        """Score `input_path` against every registered original.

        Returns a list of dicts, one per target, of the form:
            {"target": stem, "confidence": float, "evidence": [rule dicts]}
        """
        results = []
        for stem, target in self.targets.items():
            evidence = []
            total = 0
            max_score = 0
            for name in self.rule_names:
                ev = self._resolve_rule(name)(target, input_path)
                total += int(ev.get("score", 0))
                max_score += int(ev.get("out_of", 0))
                evidence.append(ev)
            confidence = round(100.0 * total / max_score, 1) if max_score else 0.0
            results.append({"target": stem, "confidence": confidence,
                            "evidence": evidence})
        return results

    def find_best_match(self, input_path):
        """Best match verdict for one suspect image.

        Returns dict with keys: input, target (original stem or None if
        REJECTED), confidence, evidence (best target's rule evidence).
        """
        results = self.evaluate(input_path)
        best = max(results, key=lambda r: r["confidence"])
        verdict = {
            "input": os.path.basename(input_path),
            "evidence": best["evidence"],
            "confidence": best["confidence"],
            "rejected": best["confidence"] < MATCH_THRESHOLD,
        }
        verdict["target"] = None if verdict["rejected"] else best["target"]
        return verdict