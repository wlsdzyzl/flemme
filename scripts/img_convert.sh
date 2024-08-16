#!/bin/bash
for i in `ls path/to/datasets/REFUGE2/train/mask/*.bmp`; do
    echo convert "$i" "${i%.bmp}.png"
    convert "$i" "${i%.bmp}.png"
done