#!/bin/bash
out="src/moon.h"
touch $out
out=$(realpath $out)

echo "#pragma once" > $out
cd src/asset
for file in moon*.png; do
    xxd -i $file >> $out
    echo "" >> $out
done