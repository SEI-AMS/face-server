#!/bin/sh
cd /home/cloudlet/face_server
LD_LIBRARY_PATH=`pwd`/lib ./FaceRec at_test.txt name_map.txt
