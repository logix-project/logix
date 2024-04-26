## Checklist
- [ ] Check `pipeline.py` and `task.py` are correctly implemented.
- [ ] Run `test_pipeline.py`.
- [ ] Run `scripts/find_hyperparameters.py` to double-check the same hyperparameters are chosen.
- [ ] Run `scripts/train.py` to train the model.
- [ ] Run `scripts/train_with_subsets.py` to train the model with subsets of data points.
- [ ] Run `scripts/check_consistency.py` to make sure the code is reproducible.
- [ ] Run `scripts/compute_lso_scores.py` to obtain the LSO scores.
- [ ] Upload the directory `files/` to Dropbox.

## Hardware & Cluster
GPU: `NVIDIA A100`.
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.41.03              Driver Version: 530.41.03    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB           On | 00000000:01:00.0 Off |                    0 |
| N/A   31C    P0               61W / 500W|      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
```

Commands:
```bash
srun -p ml -q ml -A ml -w overture -c 16 --mem=80G --gres=gpu:1 --pty bash

srun -p ml -q ml -A ml -w quartet2 -c 16 --mem=48G --gres=gpu:1 --pty bash
srun -p ml -q ml -A ml -w quartet3 -c 16 --mem=48G --gres=gpu:1 --pty bash
srun -p ml -q ml -A ml -w quartet4 -c 16 --mem=48G --gres=gpu:1 --pty bash
srun -p ml -q ml -A ml -w sonata2 -c 16 --mem=48G --gres=gpu:1 --pty bash

. /mfs1/$USER/envs/influence_research_trak_env

. /mfs1/$USER/envs/influence_research_env
cd /mfs1/$USER/code/influence-research/experiments/glue
```

## Checkpoints & LSO Scores
Use the following commands to download all files from the CS cluster:
```bash
ssh -L60000:slurm.ais.sandbox:22 jbae@cs.toronto.edu  # Keep this port open.
scp -P -r 60000 jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files /Users/$USER/code/influence-research/experiments/glue
```
Alternatively, you can use the `rsync`:
```bash
ssh -L60000:slurm.ais.sandbox:22 jbae@cs.toronto.edu
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/checkpoints /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/lso_scores /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/emb_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/prototype_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/unif_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/ensemble_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/corruption_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/ensemble_unif_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/brittleness_results /Users/$USER/code/influence-research/experiments/glue/files
rsync -azvh -e 'ssh -p 60000' jbae@localhost:/mfs1/jbae/code/influence-research/experiments/glue/files/brittleness_results_v2 /Users/$USER/code/influence-research/experiments/glue/files


```