python gener_depth_reconstructer_v3.py \
  --data_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100 \
  --depth_source raw \
  --raw_depth_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100/depth \
  --out_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100/output/reconstruction3_raw \
  --frame_stride 1 \
  --pcd_voxel_size 0.03 \
  --tsdf_voxel_length 0.02 \
  --sdf_trunc 0.06

python gener_depth_reconstructer_v3.py \
  --data_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100 \
  --depth_source predicted \
  --depth_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100/output_stride_1/depth \
  --out_dir /home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100/output/reconstruction3_predicted \
  --frame_stride 1 \
  --pcd_voxel_size 0.03 \
  --tsdf_voxel_length 0.02 \
  --sdf_trunc 0.06
