#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:01:13 2025

@author: alex
"""

import sleap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#mpl.style.use("seaborn-deep")
sleap.versions()

#help(sleap.load_metrics)

model_path = '/Users/alex/Downloads/250729_123240.single_instance.n=67'
#metrics = sleap.load_metrics("td_fast.210505_012601.centered_instance.n=1800", split="val")
metrics = sleap.load_metrics(model_path, split="val")
print("\n".join(metrics.keys()))

print("Error distance (50%):", metrics["dist.p50"])
print("Error distance (90%):", metrics["dist.p90"])
print("Error distance (95%):", metrics["dist.p95"])

#Plot histogram of localization error
plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
sns.histplot(metrics["dist.dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability")
plt.xlabel("Localization error (px)");
plt.show()

#Plot OKS Scores
plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
sns.histplot(metrics["oks_voc.match_scores"].flatten(), binrange=(0, 1), kde=True, kde_kws={"clip": (0, 1)}, stat="probability", bins=10)
plt.xlabel("Object Keypoint Similarity");
plt.show()

plt.figure(figsize=(4, 4), dpi=150, facecolor="w")
for precision, thresh in zip(metrics["oks_voc.precisions"][::2], metrics["oks_voc.match_score_thresholds"][::2]):
    plt.plot(metrics["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left");
plt.show()

#Want these to be close to 1
print("mAP:", metrics["oks_voc.mAP"])
print("mAR:", metrics["oks_voc.mAR"])