from hyperassist import parameter_assist, log_assist
import numpy as np
import time 

# log_assist.process() 

print("Test run: Log check")
log_assist.process("./example.log")

print("\n\n ------------ \n\n ")
print("Test run: Live log check")

log_assist.live()
for epoch in range(1, 6):
    print(f"epoch: {epoch} | train_loss: {1.0/epoch:.3f} | val_loss: {1.2/epoch:.3f} | acc: {0.5 + 0.1*epoch:.2f} | lr: 0.001 | grad_norm: {0.9+0.1*epoch:.2f}")
    if epoch == 3:
        print("epoch: 3 | train_loss: nan | val_loss: nan | acc: 0.70 | lr: 0.001 | grad_norm: 10.0")
    time.sleep(0.1)
print("Finished dummy training.")
log_assist.summarize_live()

print("\n\n ------------ \n\n ")
print("Test run: Normal hyperparameter check")

params = {
    "learning_rate": 0.0005,
    "weight_decay": 0.01,
    "dropout": 0.2,
    "optimizer": "adamw",
    "batch_size": 64,
    "epochs": 20,
}

# Check #1 normal check 

parameter_assist.check(
    params,
    model_type="transformer",           # type of model (for recommendations)
    dataset_size=100000,                # number of samples
    cpu_cores=8,                        # from system or user
    ram_gb=32,                          # from system or user
    num_gpus=2,                         # user sets, not auto-detected!
    per_device_batch=32,                # sometimes used for multi-GPU
    num_devices=2,                      # usually = num_gpus
    parameter_budget=350_000_000,       # total model params (optional, advanced)
    hidden_size=1024,                   # context for architecture formulas
    num_layers=24,                      # context for architecture formulas
    buffer_factor=1.5,                  # for batch size recs
    max_batch=256,                      # for batch size recs
    bytes_per_sample=4096,              # memory estimation
    epochs=20,                          # OK to repeat (just context)
    risk_level="medium",                # for dropout/attention advice
    method="kaiming_uniform",           # for param init advice
    multiple_of=128,                    # for hidden size recs
    datasets_file="test.json",          # or datasets_folder=...
    warmup_pct=0.05,                    # for warmup step recs
    depth=24,                           # for embedding advice
    vocab_size=32000,                   # for NLP tasks
)

print("\n\n ------------ \n\n ")
print("Test run: Advance hyperparameter check")
# Check #2 advance check

fixed_preds = np.array([0, 1, 0, 1, 1])
fixed_labels = np.array([0, 0, 1, 1, 1])

empirical_loss_func = lambda p: float(np.mean(fixed_preds != fixed_labels))
kl_func = lambda p: (1 - p) * 100000
parameter_assist.check(
    params,
    model_type="transformer",
    dataset_size=100000,
    cpu_cores=8,
    ram_gb=32,
    num_gpus=2,
    per_device_batch=32,
    num_devices=2,
    parameter_budget=350_000_000,
    hidden_size=1024,
    num_layers=24,
    buffer_factor=1.5,
    max_batch=256,
    bytes_per_sample=4096,
    risk_level="medium",
    method="kaiming_uniform",
    multiple_of=128,
    datasets_file="test.json",
    warmup_pct=0.05,
    depth=24,
    vocab_size=32000,
    train_loss=1.23,
    theory="on",
    epochs=20,

    # L4 and all missing args (dummy/test values, replace as needed!)
    # Skipped: estimate_epoch_count
    total_steps=31250,

    # recommend_weight_decay
    compute_factor=2.0,

    # recommend_batch_size
    # (already included: compute_factor above)

    # recommend_graditent_accum_steps
    effective_batch_size=512,

    # recommend_param_init_scale
    fan_in=4096,

    # recommend_warmup_steps
    # (already included: total_steps, warmup_pct above)

    # recommend_dropout_rate
    model_complexity=7.5,

    # recommend_transformer_learning_rate
    step=20000,
    d_model=1024,
    warmup_steps=4000,

    # recommend_curvature_aware_weight_decay
    hessian_max_eig=1.2,
    alpha=0.1,
    min_weight_decay=1e-5,

    # recommend_entropy_init_scale
    loss_entropy=2.0,
    weight_var=0.5,
    epsilon=1e-8,

    # recommend_fisher_ntk_lr
    fisher_trace=50.0,
    ntk_max_eig=3.1,
    lr_scale=1.0,
    # epsilon=1e-8, (already included above)

    # recommend_gns_batch_size
    gradient_noise_scale=0.7,
    target_variance=1.0,
    min_batch_size=16,

    # recommend_ib_attention_dropout
    I_TX=1.1,
    I_TY=0.9,
    beta=0.3,
    dropout_max=0.4,
    dropout_min=0.1,
    # epsilon=1e-8, (already included above)

    # recommend_pacbayes_dropout_rate_advanced
    N=100000,
    n=10000,
    delta=0.05,
    kl_func=kl_func,
    empirical_loss_func=empirical_loss_func,
    p_bounds=(0.0, 1.0),
    c=1.0,
    penalty_scale=0.1,
    min_penalty=1e-3,
)

