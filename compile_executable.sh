#!/usr/bin/env bash

set -e
arg_one=$1
shift

cargo run --release -- -n $@ "$arg_one"

g++ -O3 -L../target/release/ "$arg_one.o" -lbfrt -o $(basename "${arg_one%%.*}")
