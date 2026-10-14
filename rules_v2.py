"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Rules 1-3 are kept exactly as in `rules.py`. Your job is to ADD Rule 4 that
targets the systematic weakness you diagnosed from results_v1_hard.txt.

Constraint: the rules' `out_of` weights must still sum to 100 so the Final
Score stays on a /100 scale (the format validator enforces "Final Score:
<s>/100" equals the sum of the block's rule scores).

Example split that keeps the original spirit: Rules 1-3 -> 25/25/40 and
Rule 4 -> 10 (or trim the losers more aggressively).
"""

import rules
import numpy as np
import cv2

#: Import the original rules and append your new one.
RULES = (
    "rule1_metadata_v2",
    "rule2_histogram_v2",
    "rule3_template",
    "rule4_edges",
)


def _rescale_rule(rule_fn, target, input_path, new_weight):

    result = rule_fn(target, input_path)
    old_weight = result["out_of"]

    result = result.copy()
    result["out_of"] = new_weight
    result["score"] = (
        int(round(new_weight * result["score"] / old_weight))
        if old_weight else 0 
    )
    return result 

def rule1_metadata_v2(target, input_path):
    return _rescale_rule(rules.rule1_metadata, target, input_path, 25)

def rule2_histogram_v2(target, input_path):
    return _rescale_rule(rules.rule2_histogram, target, input_path, 25)

def rule4_edges(target, input_path):

    out =  {
        "rule": 4,
        "name": "Edges",
        "fired": False,
        "score": 0,
        "out_of": 10,
        "note": "Edge similarity 0.00",
        "metric": 0.0,
    }
    try:
       original = cv2.imread(target["path"], cv2.IMREAD_GRAYSCALE)
       suspect = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

       if original is None or suspect is None:
           return out 

       def prepare(image):
           h, w = image.shape
           scale = min(1.0, 512 / max(h, w))
           if scale < 1.0:
              image = cv2.resize(
                  image, (int(w * scale), int(h * scale))
              )
           return cv2.Canny(image, 100, 200)

       original_edges = prepare(original)
       suspect_edges = prepare(suspect)

       oh, ow = original_edges.shape
       sh, sw = suspect_edges.shape
       best = 0.00

       for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
           tw, th = int(sw * scale), int(sh * scale)
   
           if tw < 20 or th < 20 or tw > ow or th > oh:
               continue

           template = cv2.resize(suspect_edges, (tw, th))

           if np.std(template) < 1.0:
               continue

 
           result = cv2.matchTemplate(
               original_edges, template, cv2.TM_CCOEFF_NORMED
           )
           score = float(np.nan_to_num(result.max(), nan=0.00))
           best = max(best, score)

       metric = max(0.0, min(1.0, best))
       out["metric"] = round(metric, 3)
       out["note"] = f"Edge similarity {metric:.2f}"

       if metric >= 0.4:
           out["fired"] = True 
           out["score"] = int(round(10 * metric))
     
    except Exception as exc:
        out["note"] = f"Edge check error: {type(exc).__name__}: {exc}"
  
    return out 
