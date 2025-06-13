from pathlib import Path
import glob
import torch
import numpy as np
from models import *

from utils.helper_funcs import (
    load_config,
    save_sampling_results_as_imgs,
    get_model_path,
    get_conf_name,
    print_config,
    draw_boundary,
    mean_of_list_of_tensors,
)
from forward.forward_schedules import ForwardSchedule
from reverse.reverse_process import sample
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules.transforms import DiffusionTransform
import torchvision
import json
from PIL import Image
import shutil
from torch.utils.tensorboard import SummaryWriter
from common.logging import get_logger
from argument import get_argparser, sync_config
import sys, os
from metrics import get_binary_metrics
from loaders.dataloaders import get_dataloaders
import warnings
warnings.filterwarnings("ignore")


# ------------------- params --------------------
argparser = get_argparser()
args = argparser.parse_args(sys.argv[1:])

config = load_config(args.config_file)
config = sync_config(config, args)

logger = get_logger(
    filename=f"{config['model']['name']}_test", dir=f"logs/{config['dataset']['name']}"
)
print_config(config, logger)


jet = plt.get_cmap("jet")


def write_imgs(
    imgs, prds, mid_prds, step, id, dataset, ids=None
):
    # Normalize images to [0, 1]
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    img_grid = torchvision.utils.make_grid(imgs)
    prd_grid = torchvision.utils.make_grid(prds)

    # Convert mid predictions to a color map (jet)
    mid_prds_jet = torch.zeros_like(imgs)
    for i, mid_prd in enumerate(mid_prds.detach().cpu().numpy()):
        t = jet(mid_prd[0]).transpose(2, 0, 1)[:-1, :, :]  # Remove alpha channel from the colormap
        t = np.log(t + 0.1)  # Apply log scale for better visualization
        t = (t - t.min()) / (t.max() - t.min())  # Normalize again
        mid_prds_jet[i, :, :, :] = torch.tensor(t)

    mid_prd_grid = torchvision.utils.make_grid(mid_prds_jet)

    # Draw boundary on predictions (red color)
    res_grid = draw_boundary(torch.where(prd_grid > 0, 1, 0), img_grid, (0, 0, 255))

    # Concatenate images, mid predictions, and prediction boundary grid for visualization
    img_msk_prd_grid = torch.concat(
        [
            img_grid,
            mid_prd_grid,
            torch.tensor(res_grid).to(device),
        ],
        dim=1,
    )

    # Log the results as images using TensorBoard
    writer.add_image(f"{dataset}/Test:{id}", img_msk_prd_grid, step)



writer = SummaryWriter(f'{config["run"]["writer_dir"]}/{config["model"]["name"]}')

timesteps = config["diffusion"]["schedule"]["timesteps"]
epochs = config["training"]["epochs"]
INPUT_SIZE = config["dataset"]["input_size"]

batch_size = config["data_loader"]["train"]["batch_size"]
img_channels = config["dataset"]["img_channels"]
msk_channels = config["dataset"]["msk_channels"]

ensemble = config["testing"]["ensemble"]

ID = get_conf_name(config)
device = torch.device(config["run"]["device"])
# device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device is <{device}>")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# --------- check required dirs --------------------
Path(config["model"]["save_dir"]).mkdir(exist_ok=True)
if config["testing"]["result_imgs"]["save"]:
    try:
        Path(config["testing"]["result_imgs"]["dir"] + "/" + ID).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        shutil.rmtree(Path(config["testing"]["result_imgs"]["dir"] + "/" + ID))
        Path(config["testing"]["result_imgs"]["dir"] + "/" + ID).mkdir()



forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])
DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))

# --------------- Datasets and Dataloaders -----------------
te_dataloader = get_dataloaders(config, "te")
val_dataloader = get_dataloaders(config, "vl")
if config["testing"]["dataset"] == "vl":
    dataloader = val_dataloader
else:
    dataloader = te_dataloader
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


Net = globals()[config["model"]["class"]]
model = Net(**config["model"]["params"])
if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = torch.nn.DataParallel(model)
model.to(device)


# ------------------------ EMA -------------------------------
# https://github.com/lucidrains/ema-pytorch
from ema_pytorch import EMA
try:
    if config["training"]["ema"]["use"]:
        print("Using EMA")
        ema = EMA(model=model, **config["training"]["ema"]["params"])
        ema.to(device)
    else:
        ema = None
except KeyError:
    logger.exception("You need to determine the EMA parameters at <config.training>!")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# if config["testing"]["model_weigths"]["overload"]: 
#     logger.info(f"trying to load the desired model:")
#     best_model_path = config["testing"]["model_weigths"]["file_path"]
# else:
#     logger.info(f"trying to load the best model:")
#     best_model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
best_model_path = args.best_model_path if args.best_model_path else get_model_path(name=ID, dir=config["model"]["save_dir"])

logger.info(f" -> {best_model_path}")
if not os.path.isfile(best_model_path):
    logger.exception(f"wanted to load {best_model_path} file but it does not exist!")

checkpoint = torch.load(best_model_path, map_location="cpu")
try:
    if ema:
        ema.load_state_dict(checkpoint["ema"])
        model = ema.ema_model
        logger.info("ema loaded...")
    else:
        model.load_state_dict(checkpoint["model"])
        logger.info("simple model loaded...")
except:
    logger.exception("Something happened on loading the model weights!")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Number of model parameters: {total_params}")
model.eval()


test_metrics = get_binary_metrics()

for step, batch in tqdm(
    enumerate(dataloader),
    desc=f"Testing {config['model']['name']}",
    total=len(dataloader),
):
    batch_imgs = batch["image"].to(device)
    batch_ids = batch["id"]

    samples_list, mid_samples_list = [], []
    all_samples_list = []
    for en in range(ensemble):
        samples = sample(
            forward_schedule,
            model,
            images=batch_imgs,
            out_channels=1,
            desc=f"ensemble {en+1}/{ensemble}",
        )
        samples_list.append(samples[-1][:, :1, :, :].to(device))
        mid_samples_list.append(
            samples[-int(0.1 * timesteps)][:, :1, :, :].to(device)
        )
        all_samples_list.append([s[:, :1, :, :] for s in samples])

    # preds = samples[-1].to(device)
    preds = mean_of_list_of_tensors(samples_list)
    mid_preds = mean_of_list_of_tensors(mid_samples_list)

    write_imgs(
        batch_imgs,
        preds,
        mid_preds,
        step=step,
        id=f"{ID}_BV_E{ensemble}",
        dataset=config["dataset"]["name"].upper()
    )
    if config["testing"]["result_imgs"]["save"]:
        save_sampling_results_as_imgs(
            batch_imgs,
            batch_ids,
            preds,
            all_samples_list,
            middle_steps_of_sampling=8,
            save_dir=config["testing"]["result_imgs"]["dir"],
            dataset_name=config["dataset"]["name"].upper(),
            result_id=f"{ID}_BV_E{ensemble}",
            img_ext="png",
            save_mat=True,
        )

# result = test_metrics.compute()
# writer.add_scalars(
#     f"Metrics/test-s{INPUT_SIZE}/{ID}_BV_E{ensemble}",
#     result,
# )

# logger.info(f"result for best model {ID}-E{ensemble}")
# logger.info(json.dumps({k: v.item() for k, v in result.items()}, indent=4))