#!/bin/bash
#SBATCH -J a10_27708473_W
#SBATCH -p mit_normal
#SBATCH -t 00:12:00
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -D /orcd/pool/008/so3_shared/marielle/projects/.worktrees/xtrax/261102-a1-a10-probes
#SBATCH -o /orcd/pool/008/so3_shared/marielle/projects/.worktrees/xtrax/261102-a1-a10-probes/outputs/probes/a10/raw/a10_27708473_W_%j.log
#SBATCH --requeue
exec /usr/bin/python3.11 scripts/experiments/a10_writer.py --segdir /orcd/scratch/orcd/011/maarxaru/a10_probe --duration-s 240
