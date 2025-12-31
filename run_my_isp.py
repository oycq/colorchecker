#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import my_isp


def main():
    from pathlib import Path

    inputs = sorted(Path("scene").rglob("*raw.png"))
    for p in inputs:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img = raw.astype(np.float32) / 65535.0
        out = my_isp.process(img)
        cv2.imwrite(str(p.parent / "my_isp.png"), out)
        print(p)


if __name__ == "__main__":
    main()