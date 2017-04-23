#!/bin/bash

while :
do
  EVENT=$(inotifywait --format '%e' weights.hdf5)
  [ $? != 0 ] && exit
  [ "$EVENT" = "MODIFY" ] && git commit -a -m 'new optimal weights found' && \
    git push
done
