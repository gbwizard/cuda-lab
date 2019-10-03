#!/bin/bash

CURDIR=$PWD
THISDIR=$(readlink -f $(dirname $0))
QDDIR=$THISDIR

cd $QDDIR
OUT=`find . -maxdepth 1 -iname *.files | tail -1`
[ "$OUT" ] || { echo "Error: no *.files files found in this directory. Bailout"; exit 1; }

EXCLUDE=" \
    ./.git \
    ./_build \
    "

INCLUDE=" \
    "

for DIR in ./Built*; do
    EXCLUDE="$EXCLUDE $DIR"
done

PRUNE_ARGS=
for arg in $EXCLUDE; do
    if [ "$PRUNE_ARGS" ]; then
        PRUNE_ARGS="-path $arg -prune -o $PRUNE_ARGS"
    else
        PRUNE_ARGS="-path $arg -prune"
    fi
done

find . -type f -print $PRUNE_ARGS > $OUT && echo "OK ;)" || echo "Failed :("

INCLUDE_ARGS=
for arg in $INCLUDE; do
    find $arg -type f -print >> $OUT
done

cd $CURDIR
