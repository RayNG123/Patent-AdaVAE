GPU=0
MODE=interpolate # or analogy
EXPERIMENT="claim_data_iter22000_as128_scalar1.0_cycle-auto_prenc-start_wsTrue_lg-latent_attn_add_attn_beta1.0_reg-kld_attn_mode-none_ffn_option-parallel_ffn_enc_layer-8_dec_layer-12_zdim-32_optFalse_ftFalse_zrate-0.5_fb-1sd-42_10.27"

CUDA_VISIBLE_DEVICES=$GPU python test.py \
        --experiment $EXPERIMENT \
        --mode $MODE \
        --weighted_sample \
        --add_attn --latent_size 32 \
        --max_length 20 \
        --batch_size 10