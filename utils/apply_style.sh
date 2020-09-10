#!/bin/bash
# yapf==0.30.0
root_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )
yapf --in-place -vv -r --style google "$root_dir/"
