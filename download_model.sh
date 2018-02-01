#!/bin/sh

set -e

cd $(dirname $0)

usage() {
  echo "Usage $0 <version>"
  echo "available versions are 2, 3, 4, 5"
}

version=$1

if [ 0"$version" -ge 2 -a 0"$version" -le 5 ] ; then
  echo "now downloading challenge $version model"
else
  usage
  exit
fi

name="challenge$version"

mkdir -p data/model/

curl -L https://raw.githubusercontent.com/mokemokechicken/reversi-alpha-zero-models/master/${name}/model_config.json -o data/model/model_best_config.json
curl -L https://raw.githubusercontent.com/mokemokechicken/reversi-alpha-zero-models/master/${name}/model_weight.h5 -o data/model/model_best_weight.h5

echo OK
