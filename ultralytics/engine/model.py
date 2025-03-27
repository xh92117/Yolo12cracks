# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.engine.results import Results  # 直接导入Results类
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    TESTS_RUNNING,
    callbacks,
    checks,
    emojis,
    yaml_load,
)

# 直接定义TASK2DATA，避免从cfg导入
TASK2DATA = {
    "detect": "coco128.yaml",
    "segment": "coco128-seg.yaml",
    "classify": "imagenet100.yaml",
    "pose": "coco128-pose.yaml",
    "detect_cracks": "highway-cracks-6/data.yaml"
}

def get_save_dir(args, name=None):
    """获取保存目录的路径"""
    if isinstance(args, dict):
        if "save_dir" in args:
            save_dir = args["save_dir"]
        else:
            from ultralytics.utils.files import increment_path
            project = args.get("project", "") or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.get("task", "")
            name = name or args.get("name", "") or f"{args.get('mode', '')}"
            save_dir = increment_path(Path(project) / name, exist_ok=args.get("exist_ok", False) if RANK in {-1, 0} else True)
    else:
        if getattr(args, "save_dir", None):
            save_dir = args.save_dir
        else:
            from ultralytics.utils.files import increment_path
            project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
            name = name or args.name or f"{args.mode}"
            save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)

def get_cfg(cfg, overrides=None):
    """加载配置"""
    from ultralytics.cfg import get_cfg as _get_cfg
    return _get_cfg(cfg, overrides)

# 导入HUBTrainingSession和Results
def _hub_ops():
    from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
    return HUB_WEB_ROOT, HUBTrainingSession

def _get_engine_classes():
    from ultralytics.engine.results import Results
    return Results

class Model(nn.Module):
    """
    A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training,
    validation, prediction, exporting, and benchmarking. It handles different types of models, including those
    loaded from local files, Ultralytics HUB, or Triton Server.

    Attributes:
        callbacks (Dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (Dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (Dict): A dictionary of overrides for model configuration.
        metrics (Dict): The latest training/validation metrics.
        session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of
        model sources, including local files, Ultralytics HUB models, and Triton Server models. The method
        initializes several important attributes of the model and prepares it for operations like training,
        prediction, or export.

        Args:
            model (Union[str, Path]): Path or name of the model to load or create. Can be a local file path, a
                model name from Ultralytics HUB, or a Triton Server model.
            task (str | None): The task type associated with the YOLO model, specifying its application domain.
            verbose (bool): If True, enables verbose output during the model's initialization and subsequent
                operations.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = Model("path/to/model.yaml", task="detect")
            >>> model = Model("hub_model", verbose=True)
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = {}  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        self.model_name = model

        # 定义任务映射
        self._task_map = {
            'detect': {
                'model': 'DetectionModel',
                'trainer': 'DetectionTrainer',
                'validator': 'DetectionValidator',
                'predictor': 'DetectionPredictor'
            },
            'segment': {
                'model': 'SegmentationModel',
                'trainer': 'SegmentationTrainer',
                'validator': 'SegmentationValidator',
                'predictor': 'SegmentationPredictor'
            },
            'classify': {
                'model': 'ClassificationModel',
                'trainer': 'ClassificationTrainer',
                'validator': 'ClassificationValidator',
                'predictor': 'ClassificationPredictor'
            },
            'pose': {
                'model': 'PoseModel',
                'trainer': 'PoseTrainer',
                'validator': 'PoseValidator',
                'predictor': 'PosePredictor'
            },
            'detect_cracks': {
                'model': 'CrackDetectionModel',
                'trainer': 'DetectionTrainer',
                'validator': 'DetectionValidator',
                'predictor': 'DetectionPredictor'
            }
        }

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            # Fetch model from HUB
            checks.check_requirements("hub-sdk>=0.0.12")
            _, HUBTrainingSession = _hub_ops()
            session = HUBTrainingSession.create_session(model)
            model = session.model_file
            if session.train_args:  # training sent from HUB
                self.session = session

        # Check if Triton Server model
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # set `task=detect` if not explicitly set
            return

        # Load or create new YOLO model
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        # Delete super().training for accessing self.model.training
        del self.training

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Alias for the predict method, enabling the model instance to be callable for predictions.

        This method simplifies the process of making predictions by allowing the model instance to be called
        directly with the required arguments.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source of
                the image(s) to make predictions on. Can be a file path, URL, PIL image, numpy array, PyTorch
                tensor, or a list/tuple of these.
            stream (bool): If True, treat the input source as a continuous stream for predictions.
            **kwargs: Additional keyword arguments to configure the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """
        Checks if the given model string is a Triton Server URL.

        This static method determines whether the provided model string represents a valid Triton Server URL by
        parsing its components using urllib.parse.urlsplit().

        Args:
            model (str): The model string to be checked.

        Returns:
            (bool): True if the model string is a valid Triton Server URL, False otherwise.

        Examples:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolov8n")
            True
            >>> Model.is_triton_model("yolo11n.pt")
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """
        Check if the provided model is an Ultralytics HUB model.

        This static method determines whether the given model string represents a valid Ultralytics HUB model
        by checking if it contains specific URL patterns related to Ultralytics HUB.

        Args:
            model (str): The model string to be checked.

        Returns:
            (bool): True if the model string represents an Ultralytics HUB model, False otherwise.

        Examples:
            >>> Model.is_hub_model("https://hub.ultralytics.com/models/example_model")
            True
            >>> Model.is_hub_model("yolo11n.pt")
            False
        """
        HUB_WEB_ROOT, _ = _hub_ops()
        return any(x in str(model) for x in (f"{HUB_WEB_ROOT}/models/", "https://hub.ultralytics.com/models/"))

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initializes a new model and infers the task type from the model definitions.

        This method creates a new model instance based on the provided configuration file. It loads the model
        configuration, infers the task type if not specified, and initializes the model using the appropriate
        class from the task map.

        Args:
            cfg (str): Path to the model configuration file in YAML format.
            task (str | None): The specific task for the model. If None, it will be inferred from the config.
            model (torch.nn.Module | None): A custom model instance. If provided, it will be used instead of creating
                a new one.
            verbose (bool): If True, displays model information during loading.

        Raises:
            ValueError: If the configuration file is invalid or the task cannot be inferred.
            ImportError: If the required dependencies for the specified task are not installed.

        Examples:
            >>> model = Model()
            >>> model._new("yolov8n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        Loads a model from a checkpoint file or initializes it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolov8n -> yolov8n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """
        Checks if the model is a PyTorch model and raises a TypeError if it's not.

        This method verifies that the model is either a PyTorch module or a .pt file. It's used to ensure that
        certain operations that require a PyTorch model are only performed on compatible model types.

        Raises:
            TypeError: If the model is not a PyTorch module or a .pt file. The error message provides detailed
                information about supported model formats and operations.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # No error raised
            >>> model = Model("yolov8n.onnx")
            >>> model._check_is_pytorch_model()  # Raises TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        """
        Resets the model's weights to their initial state.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True,
        enabling them to be updated during training.

        Returns:
            (Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "Model":
        """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (Union[str, Path]): Path to the weights file or a weights object.

        Returns:
            (Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # remember the weights for DDP training
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        Saves the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename. It includes metadata such as
        the date, Ultralytics version, license information, and a link to the documentation.

        Args:
            filename (Union[str, Path]): The name of the file to save the model to.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Logs or returns model information.

        This method provides an overview or detailed information about the model, depending on the arguments
        passed. It can control the verbosity of the output and return the information as a list.

        Args:
            detailed (bool): If True, shows detailed information about the model layers and parameters.
            verbose (bool): If True, prints the information. If False, returns the information as a list.

        Returns:
            (List[str]): A list of strings containing various types of information about the model, including
                model summary, layer details, and parameter counts. Empty if verbose is True.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # Prints model summary
            >>> info_list = model.info(detailed=True, verbose=False)  # Returns detailed info as a list
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers in the model for optimized inference.

        This method iterates through the model's modules and fuses consecutive Conv2d and BatchNorm2d layers
        into a single layer. This fusion can significantly improve inference speed by reducing the number of
        operations and memory accesses required during forward passes.

        The fusion process typically involves folding the BatchNorm2d parameters (mean, variance, weight, and
        bias) into the preceding Conv2d layer's weights and biases. This results in a single Conv2d layer that
        performs both convolution and normalization in one step.

        Raises:
            TypeError: If the model is not a PyTorch nn.Module.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # Model is now fused and ready for optimized inference
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Generates image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image
        source. It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): The source of the image for
                generating embeddings. Can be a file path, URL, PIL image, numpy array, etc.
            stream (bool): If True, predictions are streamed.
            **kwargs: Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> List[Results]:
        """
        Performs prediction using the YOLO model.

        This method is used for inference on various types of inputs, such as images, videos, or streams.
        It configures and runs the prediction process with the provided arguments.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple, optional): The
                source of the image to make predictions on. Accepts various formats, including file paths,
                URLs, PIL images, numpy arrays, and PyTorch tensors.
            stream (bool, optional): Whether to treat the input as a continuous stream. Defaults to False.
            **kwargs (Any): Additional keyword arguments for the prediction process, such as confidence thresholds,
                device selection, or visualization options.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            Predict with default settings:
            >>> results = model.predict('https://ultralytics.com/images/bus.jpg')

            Predict with custom confidence threshold and device:
            >>> results = model.predict(img, conf=0.5, device='cpu')

            Predict with custom image size:
            >>> results = model.predict(img, imgsz=320)

            Predict and save results with visualization:
            >>> results = model.predict(img, save=True, save_txt=True, save_conf=True)

            Stream prediction on a video:
            >>> for results in model.predict('video.mp4', stream=True):
            ...     for result in results:
            ...         boxes = result.boxes  # Get prediction boxes
            ...         print(f"Detected {len(boxes)} objects in frame")
        """
        if isinstance(source, (list, tuple)) and len(source) > 0 and all(isinstance(x, Results) for x in source):
            LOGGER.warning("WARNING ⚠️ Source is already a Results object, semantic segmentation may be better applied "
                           "to the original source. For segmentation of detection results use 'segment(model, source)' "
                           "from the Checks module instead, i.e. 'from ultralytics.checks import segment'")

        if self.predictor is None:
            self.predictor = self._smart_load("predictor")(overrides=dict(task=self.task), _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, {"task": self.task, **kwargs})
        return self.predictor.predict_cli(source=source, stream=stream) if getattr(self.predictor.args, "cli", False) else \
            self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream. Defaults to False.
            persist (bool): If True, persists trackers between different calls to this method. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

        Raises:
            AttributeError: If the predictor does not have registered trackers.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator | None): An instance of a custom validator class for
                validating the model.
            **kwargs: Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs: Any,
    ):
        """
        Benchmarks the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is
        configured using a combination of default configuration values, model-specific arguments, method-specific
        defaults, and any additional user-provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to customize the benchmarking process. These are combined with
                default configurations, model-specific arguments, and method defaults. Common options include:
                - data (str): Path to the dataset for benchmarking.
                - imgsz (int | List[int]): Image size for benchmarking.
                - half (bool): Whether to use half-precision (FP16) mode.
                - int8 (bool): Whether to use int8 precision mode.
                - device (str): Device to run the benchmark on (e.g., 'cpu', 'cuda').
                - verbose (bool): Whether to print detailed benchmark information.

        Returns:
            (Dict): A dictionary containing the results of the benchmarking process, including metrics for
                different export formats.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided.

        Args:
            **kwargs: Arbitrary keyword arguments to customize the export process. These are combined with
                the model's overrides and method defaults. Common arguments include:
                format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
                half (bool): Export model in half-precision.
                int8 (bool): Export model in int8 precision.
                device (str): Device to run the export on.
                workspace (int): Maximum memory workspace size for TensorRT engines.
                nms (bool): Add Non-Maximum Suppression (NMS) module to model.
                simplify (bool): Simplify ONNX model.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, **kwargs):
        """
        Trains the YOLO model on a given dataset.

        This method configures the training process based on the provided keyword arguments and initiates the
        training using a Trainer object. After training, it updates the model's metrics and returns them.

        Args:
            **kwargs (Any): Keyword arguments for the training process. Common parameters include:
                data (str): Path to the dataset.
                epochs (int): Number of training epochs.
                batch (int): Batch size.
                imgsz (int): Input image size.
                device (str): Training device (e.g. 'cpu', '0', '0,1,2,3').
                workers (int): Number of worker threads for data loading.
                lr0 (float): Initial learning rate.
                resume (bool): Resume training from the last checkpoint.

        Returns:
            (Dict[str, Any]): A dictionary containing training metrics, if available and the process is not distributed.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)

        Notes:
            - If no 'data' argument is provided, it defaults to a task-specific dataset.
            - Training parameters are merged from default configurations, model overrides, and provided keyword args.
            - For distributed training (DDP), metrics will be None unless the process rank is 0.
        """
        self.predictor = None  # reset predictor
        overrides = {"data": TASK2DATA.get(self.task, "")}
        if not overrides["data"]:  # if data is none
            raise ValueError("dataset not found")
        overrides.update(kwargs)
        
        # 设置保存目录
        if kwargs.get("project") or kwargs.get("name"):
            from types import SimpleNamespace
            args_ns = SimpleNamespace(
                project=kwargs.get("project", ""),
                name=kwargs.get("name", ""),
                task=self.task,
                mode="train",
                exist_ok=kwargs.get("exist_ok", False)
            )
            overrides["save_dir"] = get_save_dir(args_ns)
            
        overrides.update(model=self.model if isinstance(self.model, (str, Path)) else None)

        # Check resume
        self.trainer = self._smart_load("trainer")(overrides=overrides, _callbacks=self.callbacks)
        if kwargs.get("resume"):
            self.trainer.resume_training()

        self.trainer.add_callback("on_train_end", self._on_train_end)
        if RANK in {-1, 0}:
            self.trainer.train()
            self.metrics = getattr(self.trainer, "metrics", None)  # training metrics
            LOGGER.info(
                f"Training completed successfully ✅\n"
                f"Results saved to {colorstr('bold', self.trainer.save_dir)}\n"
                f"Predict:         yolo predict task={self.task} model={self.trainer.best} imgsz={self.trainer.args.imgsz}\n"
                f"Validate:        yolo val task={self.task} model={self.trainer.best} imgsz={self.trainer.args.imgsz}\n"
                f"Export:          yolo export model={self.trainer.best} imgsz={self.trainer.args.imgsz}\n"
                f"Docs:            https://docs.ultralytics.com/modes/train/\n"
            )
            return self.metrics
        else:
            return None

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): If True, uses Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): The number of tuning iterations to perform. Defaults to 10.
            *args: Variable length argument list for additional arguments.
            **kwargs: Arbitrary keyword arguments. These are combined with the model's overrides and defaults.

        Returns:
            (Dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(use_ray=True, iterations=20)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Model":
        """
        Applies a function to model tensors that are not parameters or registered buffers.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like
        moving the model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like
                to(), cpu(), cuda(), half(), or float().

        Returns:
            (Model): The model instance with the function applied and updated attributes.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # Move model to GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not
        initialized, it sets it up before retrieving the names.

        Returns:
            (Dict[int, str]): A dict of class names associated with the model.

        Raises:
            AttributeError: If the model or predictor does not have a 'names' attribute.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated.

        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of nn.Module.

        Returns:
            (torch.device): The device (CPU/GPU) of the model.

        Raises:
            AttributeError: If the model is not a PyTorch nn.Module instance.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model. The transforms
        typically include preprocessing steps like resizing, normalization, and data augmentation
        that are applied to input data before it is fed into the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.

        This method allows registering custom callback functions that are triggered on specific events during
        model operations such as training or inference. Callbacks provide a way to extend and customize the
        behavior of the model at various stages of its lifecycle.

        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized
                by the Ultralytics framework.
            func (Callable): The callback function to be registered. This function will be called when the
                specified event occurs.

        Raises:
            ValueError: If the event name is not recognized or is invalid.

        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.
        It resets the callback list for the specified event to an empty list, effectively removing all
        registered callbacks for that event.

        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed

        Notes:
            - This method affects both custom callbacks added by the user and default callbacks
              provided by the Ultralytics framework.
            - After calling this method, no callbacks will be executed for the specified event
              until new ones are added.
            - Use with caution as it removes all callbacks, including essential ones that might
              be required for proper functioning of certain operations.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.

        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.

        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Resets specific arguments when loading a PyTorch model checkpoint.

        This static method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict): A dictionary containing various model arguments and settings.

        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """
        Loads the appropriate module based on the model task.

        This method dynamically selects and returns the correct module (model, trainer, validator, or predictor)
        based on the current task of the model and the provided key. It uses the task_map attribute to determine
        the correct module to load.

        Args:
            key (str): The type of module to load. Must be one of 'model', 'trainer', 'validator', or 'predictor'.

        Returns:
            (object): The loaded module corresponding to the specified key and current task.

        Raises:
            NotImplementedError: If the specified key is not supported for the current task.

        Examples:
            >>> model = Model(task="detect")
            >>> predictor = model._smart_load("predictor")
            >>> trainer = model._smart_load("trainer")

        Notes:
            - This method is typically used internally by other methods of the Model class.
            - The task_map attribute should be properly initialized with the correct mappings for each task.
        """
        try:
            # 获取类名
            class_name = self._task_map[self.task][key]
            
            # 根据任务类型和类名导入相应的模块
            if self.task == 'detect_cracks':
                if key == 'model':
                    from ultralytics.models.yolo.detect_cracks.model import CrackDetectionModel
                    return CrackDetectionModel
                elif key == 'trainer':
                    from ultralytics.models.yolo.detect.train import DetectionTrainer
                    return DetectionTrainer
                elif key == 'validator':
                    from ultralytics.models.yolo.detect.val import DetectionValidator
                    return DetectionValidator
                elif key == 'predictor':
                    from ultralytics.models.yolo.detect.predict import DetectionPredictor
                    return DetectionPredictor
            else:
                # 对于其他任务，从相应的模块导入类
                module_path = f"ultralytics.models.yolo.{self.task}"
                module = __import__(module_path, fromlist=[class_name])
                return getattr(module, class_name)
                
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        Provides a mapping from model tasks to corresponding classes for different modes.

        This property method returns a dictionary that maps each supported task (e.g., detect, segment, classify)
        to a nested dictionary. The nested dictionary contains mappings for different operational modes
        (model, trainer, validator, predictor) to their respective class implementations.

        The mapping allows for dynamic loading of appropriate classes based on the model's task and the
        desired operational mode. This facilitates a flexible and extensible architecture for handling
        various tasks and modes within the Ultralytics framework.

        Returns:
            (Dict[str, Dict[str, Any]]): A dictionary where keys are task names (str) and values are
            nested dictionaries. Each nested dictionary has keys 'model', 'trainer', 'validator', and
            'predictor', mapping to their respective class implementations.

        Examples:
            >>> model = Model()
            >>> task_map = model.task_map
            >>> detect_class_map = task_map["detect"]
            >>> segment_class_map = task_map["segment"]

        Note:
            The actual implementation of this method may vary depending on the specific tasks and
            classes supported by the Ultralytics framework. The docstring provides a general
            description of the expected behavior and structure.
        """
        return self._task_map

    def eval(self):
        """
        Sets the model to evaluation mode.

        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation.

        Returns:
            (Model): The model instance with evaluation mode set.

        Examples:
            >> model = YOLO("yolo11n.pt")
            >> model.eval()
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        Enables accessing model attributes directly through the Model class.

        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            (Any): The requested attribute value.

        Raises:
            AttributeError: If the requested attribute does not exist in the model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)
            >>> print(model.task)
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)

    def _on_train_end(self, trainer):
        """
        回调函数，在训练结束时调用
        
        Args:
            trainer: 训练器对象
        """
        # 更新模型和配置
        if RANK in {-1, 0}:
            ckpt = trainer.best if trainer.best.exists() else trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(trainer.validator, "metrics", None)
