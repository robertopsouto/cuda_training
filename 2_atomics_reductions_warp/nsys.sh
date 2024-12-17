#!/bin/bash

name=${1}

nsys profile --output=nsys/${name}_%p --stats=true -t cuda --cuda-memory-usage=true ./${name} # 2>&1 | tee ${name}_%p.txt
