#!/bin/bash
inputfolder="/media/wlsdzyzl/DATA/datasets/siqi/mesh_to_skeleton/TreeDiffusion/npy"
outputfolder="/media/wlsdzyzl/DATA/datasets/siqi/mesh_to_skeleton/TreeDiffusion/output"
subfolders=$(ls "$inputfolder")

for sfolder in ${subfolders}; do
  dataset=$(ls "$inputfolder/$sfolder")
  echo   python ../pmap2xyzr.py -i "${inputfolder}/${sfolder}" -o "${outputfolder}/${sfolder}" --xyzr --skeleton --surface --no_normalized -p 0.5 --input_suffix npy
  python ../pmap2xyzr.py -i "${inputfolder}/${sfolder}" -o "${outputfolder}/${sfolder}" --xyzr --skeleton --surface --no_normalized -p 0.5 --input_suffix npy
done