# #/usr/bin
#  python scripts/run_orbbec_demo.py \
#   --data_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100 \
#   --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
#   --out_dir ./output_2
#   --frame_stride 1

 python scripts/run_orbbec_demo.py \
  --data_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100 \
  --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
  --out_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100/output_stride_1 \
  --frame_stride 1 \
  --batch_size 4 \
  --enable_classic_methods 0 \
  --enable_pointcloud 0 \
  --enable_plots 0 \
  --save_npz 0
