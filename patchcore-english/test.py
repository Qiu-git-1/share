#pre-training
#python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 -d wood mvtec /root/autodl-tmp/patchcare/mvtec

#Evaluate the pre-trained model
#python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 --save_segmentation_images "/root/autodl-tmp/patchcare/results/evaluateAnswer/bottle" patch_core_loader -p "/root/autodl-tmp/patchcare/results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_bottle/" dataset --resize 256 --imagesize 224 -d "bottle" mvtec /root/autodl-tmp/patchcare/mvtec