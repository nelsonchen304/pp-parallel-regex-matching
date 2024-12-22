#!/bin/bash

if ! command -v wget &> /dev/null; then
    echo "wget is required to run the script."
    exit 1
fi

if ! command -v gunzip &> /dev/null; then
    echo "gunzip is required to run the script."
    exit 1
fi

if [[ ! -d "dataset" ]]; then
    echo "dataset/ not found. Are you running in project root?"
    exit 1
fi

mkdir -p dataset/traffic

i=1
for weekday in monday tuesday wednesday thursday; do
    output="dataset/traffic/w4_0${i}_outside.tcpdump"
    name="week4-${weekday}"
    if [[ -f "${output}" ]]; then
        echo "'${output}' exist, skipping ${name}"
    else
        echo "Downloading ${name}"
        wget "https://archive.ll.mit.edu/ideval/data/1999/testing/week4/${weekday}/outside.tcpdump.gz"
        gunzip outside.tcpdump.gz
        mv outside.tcpdump "${output}"
    fi
    ((i++))
done

echo "done getting all data."