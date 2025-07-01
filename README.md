### command to train vqvae : 
add boolean flags if you want to set them true else leave them:

python train_vqvae.py \
--speaker_audio_paths_train '/home/mudasir/REACT/data/train/audio/speaker/*/*.wav' \
--speaker_audio_paths_val '/home/mudasir/REACT/data/val/audio/speaker/*/*.wav' \
--save_dir '/home/mudasir/REACT/REACT_SUBMISSION/save_models' \
--mean_path '/home/mudasir/REACT/mean_face.npy' \
--std_path '/home/mudasir/REACT/std_face.npy' \
--batch_size 8 \
--num_workers 0 \
--num_epochs 100 \
--sr 16000 \
--duration 8 \
--fps 30 \
--online \
--style 


### command to train predictor :

python train_predictor.py \
--speaker_audio_paths_train '/home/mudasir/REACT/data/train/audio/speaker/*/*.wav' \
--speaker_audio_paths_val '/home/mudasir/REACT/data/val/audio/speaker/*/*.wav' \
--save_dir '/home/mudasir/REACT/REACT_SUBMISSION/save_models' \
--vqvae_path '/home/mudasir/ankit/ProbTalk3D/save_models/VQVAE/audio_duration8/entire_model.pt' \
--mean_path '/home/mudasir/REACT/mean_face.npy' \
--std_path '/home/mudasir/REACT/std_face.npy' \
--batch_size 8 \
--num_workers 0 \
--num_epochs 100 \
--sr 16000 \
--duration 8 \
--fps 30 \
--resample_method 'linear_interpolation' \
--online \
--with_style \
--early_stopping \
--device cuda