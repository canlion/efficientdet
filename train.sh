

python train.py \
  --voc_path /mnt/hdd/jinwoo/sandbox_datasets/voc_download/ \
  --train_pair '[[2007, "train"], [2007, "val"], [2012, "train"], ["2007Test", "test"]]' \
  --valid_pair '[[2012, "val"]]' \
  --lr 1e-2 \
  --batch_size 4 \
  --valid_batch_size 32 \
  --epochs 300 \
  --allow_growth

