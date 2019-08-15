#!/bin/bash

#WINDOWS=( 5 10 15 20 ) 
WINDOWS=( 15 20 )
SIZES=( 150 200 )
MAX_ITER=( 15 20 )
MIN_COUNTS=( 0 1 2 )

array=(
  Vietnam
  Germany
  Argentina
)
array2=(
  Asia
  Europe
  America
)

for index in ${!WINDOWS[*]}; do 
  for sindex in ${!SIZES[*]}; do
      for mindex in ${!MAX_ITER[*]}; do
          for cindex in ${!MIN_COUNTS[*]}; do
              echo "${WINDOWS[$index]}-${SIZES[$sindex]}-${MAX_ITER[$mindex]}-${MIN_COUNTS[$cindex]}"
              sh ./train.sh ${WINDOWS[$index]} ${SIZES[$sindex]} ${MAX_ITER[$mindex]} ${MIN_COUNTS[$cindex]}
              sleep 2m
              echo 'command finished'
          done
      done
  done
done