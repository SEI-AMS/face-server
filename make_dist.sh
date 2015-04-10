#!/bin/sh
rm -r ./dist
mkdir -f ./dist
cp FaceRec ./dist
cp at_test.txt ./dist
cp name_map.txt ./dist
cp run_from_service.sh ./dist
cp face_server.conf ./dist
cp -r orl/ ./dist/
cp -r lib/ ./dist/