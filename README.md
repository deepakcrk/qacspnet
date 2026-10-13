# QA-CSPNet

Code for *Quality Aware Feature Learning for Object Detection and Scene Text
Detection and Recognition*.


## Installation

```bash
pip install -r requirements.txt
pip install -v -e .
```

## Data

```bash

python tools/data_prep/convert_cocotext.py --src path/to/cocotext.v2.json \
    --out-dir datasets/COCOText/annotations --dev-images 2000
ln -s path/to/coco2014/train2014 datasets/COCOText/train2014


python tools/data_prep/crop_cocotext_words.py --src path/to/cocotext.v2.json \
    --images path/to/coco2014/train2014 --ann-dir datasets/COCOText/annotations \
    --out datasets/COCOText_words
```

Recognition data can be label files (`<image path> <word>` per line) or LMDB folders.

## Detection

```bash
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16
python tools/eval.py  -f exps/cocotext/qacspnet_s_cocotext.py -c path/to/best_ckpt.pth -b 16 -d 1 --conf 0.001 --test
python tools/train.py -f exps/default/qacspnet_s.py -d 8 -b 64 --fp16 -o
```

Variants: `_s_`, `_m_`, `_l_`, `_x_`. Options are set in the exp file or appended
to the train command, e.g. `... --fp16 qge_type local use_gce True`.

| Option | Default | Description |
|---|---|---|
| `use_quality` | `True` | Quality branch of QGE. |
| `qge_type` | `global` | QGE quality pooling: `global` (per channel) or `local` (per channel and position). |
| `qge_local_kernel` | `7` | Window size for `qge_type local` (odd). |
| `qge_invert` | `True` | Use the inverted gate 1 - w (`False` uses w). |
| `qge_alpha` | `free` | QGE blend factor: `free` (learnable), `bounded` (learnable in 0-1) or `fixed`. |
| `qge_alpha_init` | `0.5` | Initial (or fixed) QGE blend factor. |
| `qge_reduction` | `8` | QGE FC hidden size is max(C / `qge_reduction`, 8). |
| `qge_spatial_kernel` | `7` | Kernel of the QGE spatial branch (odd). |
| `qge_gate` | `sigmoid` | QGE gate: `sigmoid` (gate in (0, 1), so each channel/position is scaled within [1 - alpha, 1]) or `free` (no sigmoid, raw scores, the scale is unconstrained). |
| `bottleneck_dilation` | `2` | Dilation of the bottleneck's depthwise path. |
| `act` | `silu` | Activation: `silu`, `relu` or `lrelu`. |
| `stem_type` | `focus` | Stride-2 stem: `focus` or `conv` (one strided convolution). |
| `stem_kernel` | `6` | Kernel of the `conv` stem (6 matches Focus in parameters and FLOPs). |
| `use_gce` | `False` | Global Context Encoder (Transformer layers) on the stride-32 backbone map. |
| `gce_layers` | `1` | GCE Transformer layers. |
| `gce_heads` | `8` | GCE attention heads. |
| `gce_ff_ratio` | `2.0` | GCE feed-forward size as a multiple of the channels. |
| `gce_dropout` | `0.0` | GCE dropout. |
| `use_afw` | `True` | AFW module. |
| `afw_type` | `mlp` | AFW weight predictor: `mlp`, `deep` (wider projection + residual LayerNorm MLP blocks, for a much bigger AFW) or `attn` (self-attention over levels). |
| `afw_layers` | `3` | AFW MLP layers, residual blocks with `deep`, or attention layers with `attn`. |
| `afw_hidden` | `None` | AFW hidden size (default: sum of level channels / 4; sum of level channels with `deep`; 2 x `afw_attn_dim` with `attn`). |
| `afw_attn_dim` | `128` | AFW token size with `attn`. |
| `afw_heads` | `4` | AFW attention heads with `attn`. |
| `afw_weight` | `softmax` | How AFW scores become level weights: `softmax` (weights sum to 1, so the strides share a fixed total), `sigmoid` (independent gate in (0, 1) per stride), or `free` (each stride scaled by 1 + score, no normalization, floor or cap). |
| `num_shared_convs` | `3` | Shared 3x3 convs in the head. |
| `head_width` | `1.0` | Head channel multiplier. |
| `target_head_flops` | `None` | Head share of total FLOPs; sets `head_width` automatically. |
| `target_head_params` | `None` | Head share of total parameters; sets `head_width` automatically. |
| `no_decay_modules` | QGE layers | Modules excluded from weight decay. |
| `output_dir` | `./outputs` | Folder for checkpoints and logs (one subfolder per experiment). |
| `seed` / `QACSP_SEED` (env) | none | Random seed, e.g. `QACSP_SEED=1 python tools/train.py ...`. |

Backbone / head budget: `depth` and `width` size the backbone and neck, `head_width` sizes
the head on its own, and `target_head_flops` or `target_head_params` sets the head's share
of the whole model (fraction in 0-1; `head_width` is then solved automatically). All of
them can be given on the command line, and the resulting split is logged when the model
is built:

```bash
# head gets 20% of the FLOPs
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16 target_head_flops 0.20
# head gets 30% of the parameters
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16 target_head_params 0.30
# head channels halved, backbone unchanged
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16 head_width 0.5
```

Ready-made component switches: `exps/cocotext/ablate_*.py`.

Examples with the optional layers:

```bash
# local QGE + Global Context Encoder (2 layers) + attention-based AFW
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16 \
    qge_type local qge_local_kernel 7 use_gce True gce_layers 2 afw_type attn afw_layers 2

# strided-conv stem, bounded QGE blend factor, SIoU box loss
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16 \
    stem_type conv stem_kernel 6 qge_alpha bounded iou_type siou

# smaller head (20% of FLOPs), 2-layer AFW MLP, dilation 3 in the bottlenecks
python tools/train.py -f exps/cocotext/qacspnet_s_cocotext.py -d 1 -b 16 --fp16 \
    target_head_flops 0.20 afw_layers 2 bottleneck_dilation 3

# evaluate with the same options used for training
python tools/eval.py -f exps/cocotext/qacspnet_s_cocotext.py -c path/to/best_ckpt.pth -b 16 -d 1 \
    --conf 0.001 --test qge_type local use_gce True gce_layers 2 afw_type attn afw_layers 2
```

## Recognition

```bash
python tools/train_textnet.py --variant s --root datasets/COCOText_words \
    --train train.txt --val dev.txt --test path/to/iiit5k_test path/to/svt_test \
    --optimizer adam --lr 1e-3 --sched cosine --augment --amp \
    --iters 38000 --batch-size 256 --output outputs/textnet_s --seed 0

python tools/train_textnet.py --variant s --root datasets/COCOText_words \
    --test path/to/iiit5k_test path/to/svt_test --eval-only --ckpt outputs/textnet_s/best.pth
```

Model options are saved to `model_config.json`, which `--eval-only` reads back.

Examples with the optional layers:

```bash
# Transformer encoder + attention decoder, local QGE, Global Context Encoder, EMA
python tools/train_textnet.py --variant s --root datasets/COCOText_words \
    --train train.txt --val dev.txt --encoder transformer --enc-layers 3 --embed-dim 256 \
    --decoder attn --dec-layers 2 --dec-dim 256 --num-heads 8 \
    --qge-type local --gce --gce-layers 1 --ema \
    --optimizer adamw --lr 5e-4 --weight-decay 0.05 --sched cosine --iters 38000 --batch-size 256 --amp

# 3-layer BLSTM + CTC with beam search, wider input, case-sensitive alphabet, attention AFW
python tools/train_textnet.py --variant s --root datasets/COCOText_words \
    --train train.txt --val dev.txt --img-size 32 128 --enc-layers 3 --hidden 256 \
    --beam-size 5 --alphabet "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
    --afw-type attn --afw-layers 2 \
    --optimizer sgd --lr 0.01 --nesterov --sched multistep --milestones 20000 30000 --iters 38000

# evaluate a checkpoint with CTC beam search (model options are read from model_config.json)
python tools/train_textnet.py --variant s --root datasets/COCOText_words \
    --test path/to/iiit5k_test path/to/svt_test --eval-only --ckpt outputs/textnet_s/best.pth --beam-size 5
```

| Option | Default | Description |
|---|---|---|
| `--variant` | `s` | Model size: `s`, `m`, `l`, `x`. |
| `--root` | `.` | Folder that data paths are relative to. |
| `--train` | | Training label files or LMDB folders. |
| `--val` | | Dev sets for checkpoint selection. |
| `--test` | | Test sets evaluated once with the selected checkpoint. |
| `--img-size` | `32 96` | Input height and width (multiples of 16). |
| `--alphabet` | `0-9a-z` | Character set, as a string or a text file (case-sensitive if it has upper case). |
| `--max-len` | `25` | Longest word used in training. |
| `--epochs` | `20` | Training epochs. |
| `--iters` | | Train by iterations instead of epochs. |
| `--eval-interval` | `2000` | Iterations between evaluations (with `--iters`). |
| `--batch-size` | `1024` | Batch size. |
| `--workers` | `8` | Data-loading workers. |
| `--amp` | off | Mixed-precision training. |
| `--augment` | off | Training augmentation. |
| `--ema` | off | Evaluate and save an exponential moving average of the weights. |
| `--ema-decay` | `0.9999` | EMA decay. |
| `--grad-clip` | `5.0` | Maximum gradient norm (0 disables clipping). |
| `--beam-size` | `1` | CTC beam width (1 is greedy decoding). |
| `--optimizer` | `rmsprop` | `sgd`, `rmsprop`, `adam`, `adamw`, `adadelta`, `adagrad`, `adamax`, `nadam`, `radam`. |
| `--lr` | `0.01` | Base learning rate. |
| `--momentum` | sgd 0.9, rmsprop 0 | Momentum for `sgd` and `rmsprop`. |
| `--nesterov` | off | Nesterov momentum for `sgd`. |
| `--weight-decay` | `0` | Weight decay. |
| `--sched` | `none` | `none`, `cosine`, `linear`, `poly`, `step`, `multistep`, `exp`, `plateau`. |
| `--warmup` | `1000` | Linear warm-up iterations (all schedules except `none`). |
| `--min-lr-ratio` | `0.01` | Final lr / `--lr` for `cosine`, `linear`, `poly`. |
| `--step-size` | `10000` | Iterations per decay for `step` and `exp`. |
| `--milestones` | `20000 30000` | Decay iterations for `multistep`. |
| `--gamma` | `0.1` | Decay factor for `step`, `multistep`, `exp`, `plateau`. |
| `--power` | `0.9` | Exponent for `poly`. |
| `--patience` | `3` | Evaluations without improvement before `plateau` decays. |
| `--no-quality` | off | Remove the QGE quality branch. |
| `--qge-type` | `global` | QGE quality pooling: `global` or `local`. |
| `--qge-local-kernel` | `7` | Window size for `--qge-type local`. |
| `--qge-normal-gate` | off | Use the gate w instead of the inverted gate 1 - w. |
| `--qge-alpha` | `free` | QGE blend factor: `free`, `bounded` or `fixed`. |
| `--qge-alpha-init` | `0.5` | Initial (or fixed) QGE blend factor. |
| `--qge-reduction` | `8` | QGE FC hidden size is max(C / r, 8). |
| `--qge-spatial-kernel` | `7` | Kernel of the QGE spatial branch (odd). |
| `--qge-gate` | `sigmoid` | `sigmoid` (scale within [1 - alpha, 1]) or `free` (unconstrained scale per channel/position). |
| `--bottleneck-dilation` | `2` | Dilation of the bottleneck's depthwise path. |
| `--act` | `silu` | Activation: `silu`, `relu` or `lrelu`. |
| `--gce` | off | Global Context Encoder on the coarsest backbone map. |
| `--gce-layers` | `1` | GCE Transformer layers. |
| `--gce-heads` | `8` | GCE attention heads. |
| `--gce-ff-ratio` | `2.0` | GCE feed-forward size as a multiple of the channels. |
| `--no-afw` | off | Remove the AFW module. |
| `--afw-type` | `mlp` | AFW weight predictor: `mlp`, `deep` (wider projection + residual LayerNorm MLP blocks) or `attn`. E.g. `--afw-type deep --afw-layers 6 --afw-hidden 1024` for a much bigger AFW. |
| `--afw-layers` | `3` | AFW MLP layers, residual blocks with `deep`, or attention layers with `attn`. |
| `--afw-hidden` | | AFW hidden size (default: sum of level channels / 4; sum of level channels with `deep`). |
| `--afw-attn-dim` | `128` | AFW token size with `attn`. |
| `--afw-heads` | `4` | AFW attention heads with `attn`. |
| `--afw-weight` | `softmax` | `softmax` (weights sum to 1), `sigmoid` (independent gate per stride), or `free` (unconstrained scale per stride). |
| `--encoder` | `blstm` | Sequence encoder: `blstm`, `transformer` or `none`. |
| `--enc-layers` | `2` | BLSTM or Transformer encoder layers. |
| `--hidden` | `256` | BLSTM hidden units per direction. |
| `--embed-dim` | `64` | Projected feature size (Transformer encoder size). |
| `--rnn-dropout` | `0.0` | Dropout between BLSTM layers. |
| `--decoder` | `ctc` | Decoder: `ctc` or `attn` (Transformer attention decoder). |
| `--dec-layers` | `2` | Attention decoder layers. |
| `--dec-dim` | `256` | Attention decoder size. |
| `--num-heads` | `4` | Attention heads in the Transformer encoder and decoder. |
| `--ff-dim` | `1024` | Transformer feed-forward size. |
| `--dropout` | `0.1` | Transformer dropout. |
| `--level-fusion` | `time` | How the three backbone levels (strides 4, 8, 16) are fused: `time` (concatenated along time), `channel` (upsampled to the stride-4 width, concatenated per position), `fpn` (on the stride-4 grid: concat + 1x1 conv, residual), `add` (sum + 3x3 conv), `asff` (per-position softmax weights over the levels, weighted sum + 3x3 conv) or `sk` (per-channel softmax weights over the levels from a pooled FC, weighted sum + 3x3 conv). |
| `--fpn-upsample` | `bilinear` | With `fpn` / `add` / `asff` / `sk`: how strides 8 and 16 reach the stride-4 grid: `bilinear`, `shuffle` (1x1 conv + pixel shuffle) or `deconv`. |
| `--height-pool` | `none` | Height to one value per column: `none` (flatten C x H), `avg`, `max` or `conv` (CRNN-style strided convs, grid fusions only). |
| `--level-dropout` | `0.0` | Training only, with `channel` / grid fusions: fraction of images with one random level zeroed. |
| `--ckpt` | | Checkpoint to load. |
| `--eval-only` | off | Evaluate `--ckpt` on `--test` without training. |
| `--resume` | | Continue from `last_state.pth`. |
| `--start-step` | `0` | Step to continue from when using `--ckpt` without `--resume`. |
| `--time-limit` | | Stop training after this many hours. |
| `--output` | `textnet_<variant>` | Output folder. |
| `--seed` | | Random seed. |
| `--metrics-out` | | JSON file for the final test accuracies. |

## Trained weights

Trained checkpoints can be downloaded from
[OneDrive](https://amritavishwavidyapeetham-my.sharepoint.com/:u:/g/personal/cr_deepak_cb_students_amrita_edu/IQDS63ekuM11Q7G-LiTTHoOdAXMwOKY7gp1tT_byRwkeUVQ?e=nOBkyU).
Pass the downloaded `.pth` file to `-c` (detection) or `--ckpt` (recognition) in the commands above.
