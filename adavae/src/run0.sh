python adaVAE.py --batch-sizes 90 --dataset yelp_data --max_length 32 --pre_enc_iter start --add_attn --add_z2adapters --beta_0 1 --fb 1 --adapter_size 128 --iterations 22000 --weighted_sample --latent_size 32 --encoder_n_layer 8 --decoder_n_layer 12 --adapter_init bert --attn_mode none  --kl_rate 0.50 &&\
python adaVAE.py --batch-sizes 90 --dataset yelp_data --max_length 32 --pre_enc_iter start --add_attn --beta_0 1 --fb 1 --adapter_size 128 --iterations 22000 --latent_gen  mean_max_linear --weighted_sample --latent_size 32 --encoder_n_layer 8 --decoder_n_layer 12 --adapter_init bert --attn_mode none  --kl_rate 0.50 &&\
python adaVAE.py --batch-sizes 90 --dataset yelp_data --max_length 32 --pre_enc_iter start --add_attn --beta_0 1 --fb 1 --adapter_size 128 --iterations 22000 --latent_gen linear --weighted_sample --latent_size 32 --encoder_n_layer 8 --decoder_n_layer 12 --adapter_init bert --attn_mode none  --kl_rate 0.50