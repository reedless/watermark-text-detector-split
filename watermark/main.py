import datetime
import logging
import os
import os.path as osp
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import cv2
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import log_every_n_seconds, setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

setup_logger()

def get_dataset_dicts(input_image_path, watermark_mask_path):

    input_image_files = sorted(os.listdir(input_image_path))

    dataset_dicts = []

    for id in tqdm(range(len(input_image_files))):
        file_id = input_image_files[id]
        input_image_file = osp.join(input_image_path, file_id)
        watermark_mask_file = osp.join(watermark_mask_path, file_id)

        if os.path.isfile(watermark_mask_file):
            watermark_mask_img = Image.open(watermark_mask_file)

            img_height, img_width = watermark_mask_img.size
            record = {"file_name": input_image_file,
                    "height": img_height,
                    "width": img_width,
                    "image_id": file_id,
                    "annotations": []}

            W = np.array(watermark_mask_img).astype(np.uint8)

            img_gray = cv2.cvtColor(W, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contour for contour in contours if contour.shape[0] > 3]

            objs = []
            for contour in contours:
                pairs = [pair[0] for pair in contour]
                px = [int(a[0]) for a in pairs]
                py = [int(a[1]) for a in pairs]
                poly = [int(p) for x in pairs for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "iscrowd": 0,
                    "category_id": 0
                }
                objs.append(obj)
            record['annotations'] += objs

        else: # if hard negative samples
            im = cv2.imread(input_image_file)
            record = {"file_name": input_image_file,
                    "height": im.shape[0],
                    "width": im.shape[1],
                    "image_id": file_id,
                    "annotations": []}

        dataset_dicts.append(record)

    return dataset_dicts

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.best_validation_loss = float("inf")

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        if mean_loss < self.best_validation_loss:
            checkpointer = DetectionCheckpointer(self._model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.save("best_model")

        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = (next_iter == self.trainer.max_iter)
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        print("test_dataset_name = {}".format(dataset_name))
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

if __name__ == '__main__':
    data_path = '../dataset'
    for d in ["train", "val"]:
        DatasetCatalog.register("watermarks_" + d,
                                lambda d=d: get_dataset_dicts(f'{data_path}/{d}/watermark_input',
                                                              f'{data_path}/{d}/watermark_mask')
                                )
        MetadataCatalog.get("watermarks_" + d).set(
            thing_classes=['watermark'])
    watermarks_metadata = MetadataCatalog.get("watermarks_train")
    watermarks_metadata_val = MetadataCatalog.get("watermarks_val")

    # Set training and validation parameters
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("watermarks_train",)
    cfg.DATASETS.TEST = ("watermarks_val",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # include image without annotation
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda"

    if sys.argv[1] == "train":
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    elif sys.argv[1] == "test":
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)

        # score_benchmark
        folder_name = 'test'
        for i in tqdm(os.listdir(f'../dataset/{folder_name}')):
            img_path = os.path.join(f'../dataset/{folder_name}', i)
            if os.path.isfile(img_path):
                im = cv2.imread(img_path)
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1],
                            metadata=watermarks_metadata,
                            scale=1,
                            instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                            )
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                fig, ax = plt.subplots(1, 2, figsize=(14, 10))
                ax[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
                ax[1].imshow(im[:, :, ::-1])

                os.makedirs(f'output/{folder_name}/', exist_ok=True)
                plt.savefig(f'./output/{folder_name}/{i}')
                plt.close()
    else:
        print("Either 'Train' or 'Test' must be specified in the argument.")
