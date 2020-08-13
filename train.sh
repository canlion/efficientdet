

python train.py \
  --voc_path /mnt/hdd/jinwoo/sandbox_datasets/voc_download/ \
  --train_pair '[[2007, "train"], [2007, "val"], [2012, "train"], ["2007Test", "test"]]' \
  --valid_pair '[[2012, "val"]]' \
  --allow_growth

