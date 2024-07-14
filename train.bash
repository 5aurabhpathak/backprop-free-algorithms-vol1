#!/bin/bash
# MIT license notice
# © 2024 Saurabh Pathak. All rights reserved
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Purpose: Training wrapper. Restarts training if crashed until 5 consecutive crashes. Also archives the code using
# which the current training will take place for future debugging

set -o errexit
set -o nounset

PROJECT_PATH="/blue"
DATA_PATH="$1"
shift

NOTIFY_PATH="$PROJECT_PATH/data/notify/"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S-%6N")

# notify path will store files that describe the status of the training and the hyper-parameters
mkdir -p $NOTIFY_PATH
mkdir -p $DATA_PATH

export PATH=$PATH:/usr/local/nvidia/bin:/usr/local/cuda/bin
export PYTHONPATH=$PATH:$PROJECT_PATH
export TF_XLA_FLAGS="--tf_xla_cpu_global_jit"
export TF_ENABLE_AUTO_GC=1

# workaround for tf.data.shuffle memory leakage issue. see here:
# https://github.com/tensorflow/tensorflow/issues/54299
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

notify() {
  local msg=$1
  shift
  NOTIFY_FILENAME="$NOTIFY_PATH/$(hostname)-status-$TIMESTAMP.txt"
  argsCopy=("$@")
  # shellcheck disable=SC2145
  printf '%s ' "$TIMESTAMP: $msg: ${argsCopy[@]}">"$NOTIFY_FILENAME"
  echo >>"$NOTIFY_FILENAME"
}

notify "running" "$@"

pushd "$DATA_PATH" || exit 1

# compress and store the code that is about to be executed. useful for debugging
[[ -f src.tar.xz ]] || tar --exclude='data' --exclude='.git' --exclude='notebooks' --exclude='__pycache__' \
-Jcf src.tar.xz "$PROJECT_PATH"

crashed_count=0

# run until completed successfully
until python "$PROJECT_PATH"/tune.py "$@"
do
    echo "Script crashed with exit code $?.  Respawning.." >&2
    sleep 1

    if [[ "$crashed_count" == 5 ]]
    then
      notify "crashed" "$@"
      exit 1
    fi

    ((crashed_count=crashed_count+1))

done

notify "completed" "$@"
exit 0
