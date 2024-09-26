from seqMNIST import *
from tqdm import tqdm
from pytorch_lightning import Trainer
from rwkv_model_x70rc2 import RwkvModel
from test_tube import Experiment
tqdm.monitor_interval = 0
AVAIL_GPUS = min(1, torch.cuda.device_count())
torch.set_float32_matmul_precision('medium')

from pytorch_lightning.loggers import TensorBoardLogger

def main():

    # input_scan_dim=28 is row-by-row sequential MNIST
    # input_scan_dim=1 to make it pixel-by-pixel (doesn't work, need to fix)
    input_scan_dim = 28
    output_dim = 10
    learning_rate = 0.0005
    batch_size = 256
    gradient_clip = 2.0
    is_permuted = False
    max_epochs = 100
    percent_validation = 0.2

    gpus = None
    if torch.cuda.is_available():
        gpus = [0]

    model = RwkvModel(input_scan_dim, output_dim)
    lightning_module = SeqMNIST(model, learning_rate, batch_size, is_permuted, percent_validation)

    logger = TensorBoardLogger("lightning_logs", name="rwkv_x70rc2", version="2layer64")
    trainer = Trainer(max_epochs=10, logger=logger, deterministic=False)
    trainer.fit(lightning_module)


if __name__ == '__main__':
    main()
