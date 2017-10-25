#!/bin/sh

set -e

cd $(dirname $0)

mkdir -p data/model/

curl -L https://raw.githubusercontent.com/mokemokechicken/reversi-alpha-zero-models/master/best_models/model_best_config.json -o data/model/model_best_config.json
curl -L https://raw.githubusercontent.com/mokemokechicken/reversi-alpha-zero-models/master/best_models/model_best_weight.h5 -o data/model/model_best_weight.h5

echo OK
