

python train.py \
  --voc_path /mnt/hdd/jinwoo/sandbox_datasets/voc_download/ \
  --train_pair '[[2007, "train"], [2007, "val"], [2012, "val"], ["2007Test", "test"]]' \
  --valid_pair '[[2012, "val"]]' \
  --freeze_backbone \
  --lr 4e-3 \
  --batch_size 16 \
  --epochs 60 \
  --allow_growth
