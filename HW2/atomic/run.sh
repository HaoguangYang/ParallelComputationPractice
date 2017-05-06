#!/bin/bash
export OMP_NUM_THREADS=8 &&make run &&mkdir threads8 && mv output* ./threads8
export OMP_NUM_THREADS=4 &&make run &&mkdir threads4 && mv output* ./threads4
