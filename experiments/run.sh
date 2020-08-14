python3 -O ../sac_x/main.py \
--num_worker=6 \
--num_grads=6 \
--update_targnets_every=300 \
--learning_steps=1200 \
--actor_lr=2e-4 \
--critic_lr=2e-4 \
--global_gradient_norm=0.5 \
--entropy_reg=1e-3 \
--replay_buffer_size=2000 \
--num_trajectories=20  \
--num_intentions=14 \
--num_observations=26
