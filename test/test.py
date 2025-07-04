import time

from hyperassist import log_assist, parameter_assist

print("=" * 40)
print(" LOG TEST 1: Analyze from STDIN")
print("=" * 40)
log_assist.process()      
print("\n" + "=" * 40)
print(" LOG TEST 2: Analyze from log file ")
print("=" * 40)
log_assist.process("./log.txt")

print("\n" + "=" * 40)
print(" LOG TEST 3: Live log capture and summary ")
print("=" * 40)
log_assist.live()
for epoch in range(1, 6):
    print(f"epoch: {epoch} | train_loss: {1.0/epoch:.3f} | val_loss: {1.2/epoch:.3f} | acc: {0.5 + 0.1*epoch:.2f} | lr: 0.001 | grad_norm: {0.9+0.1*epoch:.2f}")
    if epoch == 3:
        print("epoch: 3 | train_loss: nan | val_loss: nan | acc: 0.70 | lr: 0.001 | grad_norm: 10.0")
    time.sleep(0.1)
print("Finished dummy training.")
log_assist.summarize_live()

print("\n" + "=" * 40)
print(" PARAMETER_ASSIST TEST 1: Dict input ")
print("=" * 40)
params_dict = {
    "learning_rate": 0.05,              # Suspiciously high!
    "per_device_train_batch_size": 256, # Large batch size
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 15,
    "lr_scheduler_type": "constant"
}
parameter_assist.check(params_dict)

print("\n" + "=" * 40)
print(" PARAMETER_ASSIST TEST 2: DummyArgs input ")
print("=" * 40)
class DummyArgs:
    def __init__(self):
        self.learning_rate = 1e-7
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        self.lr_scheduler_type = "cosine"
training_args = DummyArgs()
parameter_assist.check(training_args)

print("\n" + "=" * 40)
print(" PARAMETER_ASSIST TEST 3: Auto-detect with extra info ")
print("=" * 40)
params = {
    "learning_rate": 0.01,
    "dropout": 0.7,
    "weight_decay": 1e-6,
}
parameter_assist.check(
    params,
    model_type="cnn",
    compute="medium",
    datasets_file="./test.json",  
    input_shape="3x32x32"
)

print("\n" + "=" * 40)
print(" PARAMETER_ASSIST TEST 4: Params + override info (transformer) ")
print("=" * 40)
parameter_assist.check(
    {
        "learning_rate": 0.0001,
        "dropout": 0.15,
        "weight_decay": 2e-5
    },
    model_type="transformer",
    dataset_size=20000,
    input_shape="512",
    compute="high"
)

print("\n" + "=" * 40)
print(" PARAMETER_ASSIST TEST 5: Minimal params (auto-fill) ")
print("=" * 40)
parameter_assist.check({
    "learning_rate": 0.0005,
    "dropout": 0.3,
    "weight_decay": 1e-5
})

print("\nAll tests done.")
