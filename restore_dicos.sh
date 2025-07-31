#!/bin/bash

SRC="/tiara/home/mlehmann/data/idefix-mkl/"
DEST="mlehmann@slurm-ui.twgrid.org:/ceph/sharedfs/users/m/mlehmann/idefix-mkl/"

echo "⚠️  This will overwrite all differing files, even if timestamps are the same."
echo "    From: $SRC"
echo "    To:   $DEST"
echo

# Confirm
read -p "Proceed with full overwrite? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborted."
    exit 1
fi

# Rsync with full overwrite (by checksum), excluding outputs
rsync -avzi --checksum --no-times \
    --exclude='outputs/' \
    --exclude='*.out' \
    --exclude='*.dat' \
    --exclude='submit.sh' \
    --exclude='IDEFIX-CPU.err' \
    --exclude='IDEFIX-CPU.out' \
    "$SRC" "$DEST"
