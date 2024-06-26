#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from helpers import log_format
import shutil, hashlib, random, itertools, logging, math, os, json, copy, sys

# Quiet down, you.
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

from pathlib import Path
from helpers.arguments import parse_args
from helpers.caching.memory import reclaim_memory
from helpers.legacy.validation import prepare_validation_prompt_list
from helpers.training.validation import Validation
from helpers.training.state_tracker import StateTracker
from helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager
from helpers.training.wrappers import unwrap_model
from helpers.data_backend.factory import configure_multi_databackend
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.legacy.sd_files import (
    import_model_class_from_model_name_or_path,
    register_file_hooks,
)
from helpers.training.min_snr_gamma import compute_snr
from helpers.prompts import PromptHandler
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

filelock_logger = get_logger("filelock")
connection_logger = get_logger("urllib3.connectionpool")
training_logger = get_logger("training-loop")

# More important logs.
target_level = "INFO"
# Is env var set?
if os.environ.get("SIMPLETUNER_LOG_LEVEL"):
    target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL")
logger.setLevel(target_level)
if os.environ.get("SIMPLETUNER_LOG_LEVEL"):
    training_logger_level = os.environ.get(
        "SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL", "INFO"
    )
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")


# SD 2.x specific imports
from helpers.legacy.metadata import save_model_card
from helpers.training.custom_schedule import (
    generate_timestep_weights,
    get_polynomial_decay_schedule_with_warmup,
    segmented_timestep_selection,
)
from helpers.training.model_freeze import freeze_entire_component, freeze_text_encoder


import numpy as np
import torch, diffusers, accelerate, transformers
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from helpers.legacy.pipeline import StableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    ControlNetModel,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.optimization import get_scheduler

from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers.utils import ContextManagers

tokenizer = None

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")


SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}


def compute_ids(prompt: str):
    global tokenizer
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids


def main():
    StateTracker.set_model_type("legacy")
    args = parse_args()
    StateTracker.set_args(args)

    if not args.preserve_data_backend_cache:
        StateTracker.delete_cache_files(
            preserve_data_backend_cache=args.preserve_data_backend_cache
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # Create the custom configuration
    from accelerate import InitProcessGroupKwargs
    from datetime import timedelta

    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400)
    )  # 1.5 hours
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[process_group_kwargs],
    )
    StateTracker.set_accelerator(accelerator)
    webhook_handler = None
    if args.webhook_config is not None:
        from helpers.webhooks.handler import WebhookHandler

        webhook_handler = WebhookHandler(
            args.webhook_config,
            accelerator,
            f"{args.tracker_project_name} {args.tracker_run_name}",
        )
        StateTracker.set_webhook_handler(webhook_handler)
        webhook_handler.send(
            message="SimpleTuner has launched. Hold onto your butts!",
            store_response=True,
        )

    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
    ):
        if "lora" in args.model_type:
            logger.error(
                "LoRA can not be trained with DeepSpeed. Please disable DeepSpeed via 'accelerate config' before reattempting."
            )
            sys.exit(1)
        if (
            "gradient_accumulation_steps"
            in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            args.gradient_accumulation_steps = (
                accelerator.state.deepspeed_plugin.deepspeed_config[
                    "gradient_accumulation_steps"
                ]
            )
            logger.info(
                f"Updated gradient_accumulation_steps to the value provided by DeepSpeed: {args.gradient_accumulation_steps}"
            )

    # If passed along, set the training seed now.
    if args.seed is not None and args.seed != 0:
        set_seed(args.seed, args.seed_for_each_device)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            from helpers.publishing.huggingface import HubManager

            hub_manager = HubManager(config=args)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        logger.info(
            "Enabling tf32 precision boost for NVIDIA devices due to --allow_tf32."
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.cuda.is_available():
        logger.warning(
            "If using an Ada or Ampere NVIDIA device, --allow_tf32 could add a bit more performance."
        )

    if args.lr_scale:
        logger.info(f"Scaling learning rate ({args.learning_rate}), due to --lr_scale")
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.bfloat16
    if torch.backends.mps.is_available() and "deepfloyd" in args.model_type:
        weight_dtype = torch.float32
        args.adam_bfloat16 = False
    StateTracker.set_weight_dtype(weight_dtype)

    # Load the scheduler, tokenizer and models.
    logger.info("Load tokenizer")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
    )
    if not tokenizer:
        raise Exception("Failed to load tokenizer.")

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        subfolder="scheduler",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
    )
    args.rescale_betas_zero_snr = True
    args.prediction_type = noise_scheduler.config.prediction_type
    logger.info(f"Using prediction type: {args.prediction_type}")
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )
        if args.train_text_encoder and "deepfloyd" not in args.model_type:
            text_encoder = freeze_text_encoder(
                args,
                text_encoder,
            )
        from transformers import T5EncoderModel

        if "deepfloyd" not in args.model_type:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=args.revision,
                variant=args.variant,
            )
            vae.requires_grad_(False)
        else:
            vae = None
    if webhook_handler is not None:
        webhook_handler.send(
            message=f"Loading base U-net model: `{args.pretrained_model_name_or_path}`..."
        )
    controlnet = None
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    ).to(weight_dtype)

    if not args.train_text_encoder:
        logger.info("Text encoder will remain frozen.")
        text_encoder.requires_grad_(False)

    if args.controlnet:
        if (
            "deepfloyd" in StateTracker.get_args().model_type
            or StateTracker.is_sdxl_refiner()
        ):
            raise ValueError(
                f"ControlNet is not yet supported with {'DeepFloyd' if not StateTracker.is_sdxl_refiner() else 'SDXL Refiner'} models. Please disable --controlnet, or switch to a full base model training task instead."
            )
        logger.info("Creating the controlnet..")
        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = ControlNetModel.from_pretrained(
                args.controlnet_model_name_or_path
            )
        else:
            logger.info("Initializing controlnet weights from unet")
            controlnet = ControlNetModel.from_unet(unet)
    elif "lora" in args.model_type:
        logger.info("Using LoRA training mode.")
        if webhook_handler is not None:
            webhook_handler.send(message="Using LoRA training mode.")
        # now we will add new LoRA weights to the attention layers
        # Set correct lora layers
        unet.requires_grad_(False)
        lora_initialisation_style = "gaussian"
        if hasattr(args, "lora_init_method") and args.lora_init_method is not None:
            lora_initialisation_style = args.lora_init_method
        lora_weight_init_type = (
            "gaussian"
            if torch.backends.mps.is_available()
            else lora_initialisation_style
        )
        if args.use_dora:
            lora_weight_init_type = "gaussian"
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights=lora_weight_init_type,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            use_dora=args.use_dora,
        )
        logger.info("Adding LoRA adapter to the unet model..")
        unet.add_adapter(unet_lora_config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers  # type: ignore

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.20"):
                logger.warn(
                    "SimpleTuner requires at least PyTorch 2.0.1, which in turn requires a new version (0.0.20) of Xformers."
                )
                if webhook_handler is not None:
                    webhook_handler.send(
                        message="SimpleTuner requires at least PyTorch 2.0.1, which in turn requires a new version (0.0.20) of Xformers.",
                        message_level="warning",
                    )

            unet.enable_xformers_memory_efficient_attention()
            if args.controlnet:
                controlnet.enable_xformers_memory_efficient_attention()

    if args.controlnet:
        # We freeze the base u-net for controlnet training.
        unet.requires_grad_(False)
        controlnet.train()
        controlnet.to(device=accelerator.device, dtype=weight_dtype)
        if args.train_text_encoder:
            logger.warning(
                "Unknown results will occur when finetuning the text encoder alongside ControlNet."
            )
    unet.to(device=accelerator.device, dtype=weight_dtype)

    if (
        args.freeze_unet_strategy == "bitfit"
        and not args.controlnet
        and "lora" not in args.model_type
    ):
        from helpers.training.model_freeze import apply_bitfit_freezing

        logger.info(f"Applying BitFit freezing strategy to the U-net.")
        unet = apply_bitfit_freezing(unet, args)
    elif args.freeze_unet_strategy == "bitfit" and args.controlnet:
        raise ValueError(
            "BitFit freezing strategy is not supported when training the ControlNet."
        )
    elif args.freeze_unet_strategy == "bitfit" and "lora" in args.model_type:
        raise ValueError(
            "BitFit freezing strategy is not supported when training the LoRA model, as they do not have a bias function."
        )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.controlnet:
            controlnet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Initialize the optimizer
    extra_optimizer_args = {
        "weight_decay": args.adam_weight_decay,
        "eps": args.adam_epsilon,
    }
    use_deepspeed_optimizer = False
    use_deepspeed_scheduler = False
    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
    ):
        offload_param = accelerator.state.deepspeed_plugin.deepspeed_config[
            "zero_optimization"
        ]["offload_param"]
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"][
            "offload_param"
        ]["pin_memory"] = False
        if offload_param["device"] == "nvme":
            if offload_param["nvme_path"] == "none":
                if args.offload_param_path is None:
                    raise ValueError(
                        f"DeepSpeed is using {offload_param['device']} but nvme_path is not specified."
                    )
                else:
                    accelerator.state.deepspeed_plugin.deepspeed_config[
                        "zero_optimization"
                    ]["offload_param"]["nvme_path"] = args.offload_param_path
            logger.info(
                f"Using DeepSpeed NVMe offload at {accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['offload_param']['nvme_path']}."
            )

        use_deepspeed_optimizer = True
        if "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [args.adam_beta1, args.adam_beta2],
                    "eps": args.adam_epsilon,
                    "weight_decay": args.adam_weight_decay,
                },
            }

        use_deepspeed_scheduler = True
        if "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.learning_rate,
                    "warmup_num_steps": args.lr_warmup_steps,
                },
            }
    # Initialize the optimizer
    if use_deepspeed_optimizer:
        logger.info("Using DeepSpeed optimizer.")
        optimizer_class = accelerate.utils.DummyOptim
        extra_optimizer_args["lr"] = args.learning_rate
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["eps"] = args.adam_epsilon
        extra_optimizer_args["weight_decay"] = args.adam_weight_decay
    elif args.use_prodigy_optimizer:
        logger.info("Using Prodigy optimizer. Experimental.")
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        extra_optimizer_args["lr"] = args.learning_rate
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["beta3"] = args.prodigy_beta3
        extra_optimizer_args["weight_decay"] = args.prodigy_weight_decay
        extra_optimizer_args["eps"] = args.prodigy_epsilon
        extra_optimizer_args["decouple"] = args.prodigy_decouple
        extra_optimizer_args["use_bias_correction"] = args.prodigy_use_bias_correction
        extra_optimizer_args["safeguard_warmup"] = args.prodigy_safeguard_warmup
        extra_optimizer_args["d_coef"] = args.prodigy_learning_rate
    elif args.adam_bfloat16:
        from helpers.training import adam_bfloat16

        optimizer_class = adam_bfloat16.AdamWBF16
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = args.learning_rate
    elif args.use_8bit_adam:
        logger.info("Using 8bit AdamW optimizer.")
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_class = bnb.optim.AdamW8bit
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = args.learning_rate
    elif hasattr(args, "use_dadapt_optimizer") and args.use_dadapt_optimizer:
        logger.info("Using D-Adaptation optimizer.")
        try:
            from dadaptation import DAdaptAdam
        except ImportError:
            raise ImportError(
                "Please install the dadaptation library to make use of DaDapt optimizer."
                "You can do so by running `pip install dadaptation`"
            )

        optimizer_class = DAdaptAdam
        if (
            hasattr(args, "dadaptation_learning_rate")
            and args.dadaptation_learning_rate is not None
        ):
            logger.debug(
                f"Overriding learning rate {args.learning_rate} with {args.dadaptation_learning_rate} for D-Adaptation optimizer."
            )
            args.learning_rate = args.dadaptation_learning_rate
            extra_optimizer_args["decouple"] = True
            extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
            extra_optimizer_args["lr"] = args.learning_rate

    elif hasattr(args, "use_adafactor_optimizer") and args.use_adafactor_optimizer:
        logger.info("Using Adafactor optimizer.")
        try:
            from transformers.optimization import Adafactor, AdafactorSchedule
        except ImportError:
            raise ImportError(
                "Please install the latest transformers library to make use of Adafactor optimizer."
                "You can do so by running `pip install transformers`, or, `poetry install` from the SimpleTuner directory."
            )

        optimizer_class = Adafactor
        if args.adafactor_relative_step:
            extra_optimizer_args["lr"] = None
            extra_optimizer_args["relative_step"] = True
            extra_optimizer_args["scale_parameter"] = False
            extra_optimizer_args["warmup_init"] = True
        else:
            extra_optimizer_args["lr"] = args.learning_rate
            extra_optimizer_args["relative_step"] = False
            extra_optimizer_args["scale_parameter"] = False
            extra_optimizer_args["warmup_init"] = False
    else:
        logger.info("Using AdamW optimizer.")
        optimizer_class = torch.optim.AdamW
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = args.learning_rate

    # Optimizer creation
    if (
        args.model_type == "full"
        or args.model_type == "deepfloyd-full"
        or args.model_type == "deepfloyd-stage2"
    ):
        if args.controlnet:
            params_to_optimize = (
                itertools.chain(controlnet.parameters(), text_encoder.parameters())
                if args.train_text_encoder
                else controlnet.parameters()
            )
        else:
            params_to_optimize = (
                itertools.chain(unet.parameters(), text_encoder.parameters())
                if args.train_text_encoder
                else unet.parameters()
            )
    elif (
        args.model_type == "lora"
        or args.model_type == "deepfloyd-lora"
        or args.model_type == "deepfloyd-stage2-lora"
    ):
        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
        if args.train_text_encoder:
            params_to_optimize = params_to_optimize + list(
                filter(lambda p: p.requires_grad, text_encoder.parameters())
            )

    if use_deepspeed_optimizer:
        logger.info(f"Creating DeepSpeed optimizer")
        optimizer = optimizer_class(params_to_optimize)
    else:
        logger.info(
            f"Optimizer arguments, weight_decay={args.adam_weight_decay} eps={args.adam_epsilon}, extra_arguments={extra_optimizer_args}"
        )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
        optimizer = optimizer_class(
            params_to_optimize,
            **extra_optimizer_args,
        )
    from helpers.legacy.validation import get_validation_resolutions

    # Kick out an early error for DF II trainers that used the wrong resolutions.
    get_validation_resolutions()
    # Move text_encoder to device and cast to weight_dtype)
    logging.info("Moving text encoder to GPU..")
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vae is not None:
        # Move vae, unet and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        vae_dtype = torch.bfloat16
        if hasattr(args, "vae_dtype"):
            logger.info(
                f"Initialising VAE in {args.vae_dtype} precision, you may specify a different value if preferred: bf16, fp16, fp32, default"
            )
            # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
            if args.vae_dtype == "bf16" or args.mixed_precision == "bf16":
                vae_dtype = torch.bfloat16
            elif args.vae_dtype == "fp16" or args.mixed_precision == "fp16":
                vae_dtype = torch.float16
            elif args.vae_dtype == "fp32":
                vae_dtype = torch.float32
            elif args.vae_dtype == "none" or args.vae_dtype == "default":
                vae_dtype = torch.bfloat16
        logger.debug(f"Moving VAE to GPU with {vae_dtype} precision level.")
        vae.to(accelerator.device, dtype=vae_dtype)
        logger.info(f"Loaded VAE into VRAM.")
        StateTracker.set_vae_dtype(vae_dtype)
        StateTracker.set_vae(vae)

    # Create a DataBackend, so that we can access our dataset.
    prompt_handler = None
    if not args.disable_compel:
        prompt_handler = PromptHandler(
            args=args,
            text_encoders=[text_encoder, None],
            tokenizers=[tokenizer, None],
            accelerator=accelerator,
            model_type="legacy",
        )

    try:
        if webhook_handler is not None:
            webhook_handler.send(
                message="Configuring data backends... (this may take a while!)"
            )
        configure_multi_databackend(
            args,
            accelerator,
            text_encoders=[text_encoder],
            tokenizers=[tokenizer],
            prompt_handler=prompt_handler,
        )
    except Exception as e:
        import traceback

        logging.error(f"{e}, traceback: {traceback.format_exc()}")
        if webhook_handler is not None:
            webhook_handler.send(
                message=f"Failed to load data backends: {e}", message_level="critical"
            )
        sys.exit(0)

    with accelerator.main_process_first():
        (
            validation_prompts,
            validation_shortnames,
            validation_negative_prompt_embeds,
        ) = prepare_validation_prompt_list(
            args=args, embed_cache=StateTracker.get_default_text_embed_cache()
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # Check if we have a valid gradient accumulation steps.
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )
    # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
    # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
    collected_data_backend_str = list(StateTracker.get_data_backends().keys())
    if args.push_to_hub:
        hub_manager.collected_data_backend_str = collected_data_backend_str
        hub_manager.set_validation_prompts(validation_prompts, validation_shortnames)
    logger.info(f"Collected the following data backends: {collected_data_backend_str}")
    if webhook_handler is not None:
        webhook_handler.send(
            message=f"Collected the following data backends: {collected_data_backend_str}"
        )

    total_num_batches = sum(
        [
            len(backend["metadata_backend"] if "metadata_backend" in backend else [])
            for _, backend in StateTracker.get_data_backends().items()
        ]
    )
    num_update_steps_per_epoch = math.ceil(
        total_num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None or args.max_train_steps == 0:
        if args.num_train_epochs is None or args.num_train_epochs == 0:
            raise ValueError(
                "You must specify either --max_train_steps or --num_train_epochs with a value > 0"
            )
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(
            f"Calculated our maximum training steps at {args.max_train_steps} because we have"
            f" {args.num_train_epochs} epochs and {num_update_steps_per_epoch} steps per epoch."
        )
        overrode_max_train_steps = True
    logger.info(
        f"Loading {args.lr_scheduler} learning rate scheduler with {args.lr_warmup_steps} warmup steps"
    )

    from helpers.training.custom_schedule import get_lr_scheduler

    logger.info(
        f"Loading {args.lr_scheduler} learning rate scheduler with {args.lr_warmup_steps} warmup steps"
    )
    lr_scheduler = get_lr_scheduler(
        args, optimizer, accelerator, logger, use_deepspeed_scheduler=False
    )
    if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
        lr_scheduler.num_update_steps_per_epoch = num_update_steps_per_epoch
    if hasattr(lr_scheduler, "last_step"):
        lr_scheduler.last_step = global_resume_step
    accelerator.wait_for_everyone()

    # Create EMA for the unet.
    ema_unet = None
    if args.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        if accelerator.is_main_process and webhook_handler is not None:
            webhook_handler.send(message="Creating EMA model.")
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
            decay=args.ema_decay,
        )
        logger.info("EMA model creation complete.")
    register_file_hooks(
        args,
        accelerator,
        unet,
        text_encoder,
        text_encoder_cls,
        use_deepspeed_optimizer,
        ema_unet,
        controlnet=controlnet,
    )

    train_dataloaders = []
    for _, backend in StateTracker.get_data_backends().items():
        if "train_dataloader" in backend:
            train_dataloaders.append(backend["train_dataloader"])

    if len(train_dataloaders) == 0:
        logger.error("For some reason, no dataloaders were configured.")
        sys.exit(0)

    logger.info("Preparing accelerator..")

    # Base components to prepare
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if webhook_handler is not None:
        webhook_handler.send(message="Moving weights to GPU...")
    primary_model = unet if not args.controlnet else controlnet
    results = accelerator.prepare(
        primary_model, lr_scheduler, optimizer, *train_dataloaders
    )
    if args.controlnet:
        controlnet = results[0]
    else:
        unet = results[0]
    if torch.backends.mps.is_available() or args.unet_attention_slice:
        unet.set_attention_slice("auto")
        if args.controlnet:
            controlnet.set_attention_slice("auto")

    lr_scheduler = results[1]
    optimizer = results[2]
    # The rest of the entries are dataloaders:
    train_dataloaders = results[3:]

    # Conditionally prepare the text_encoder if required
    if args.train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)
    elif args.fully_unload_text_encoder:
        del text_encoder
        text_encoder = None

    # Conditionally prepare the EMA model if required
    if args.use_ema:
        ema_unet = accelerator.prepare(ema_unet)
        logger.info("Moving EMA model weights to accelerator...")
        ema_unet.to(accelerator.device, dtype=weight_dtype)

    idx_count = 0
    for _, backend in StateTracker.get_data_backends().items():
        if idx_count == 0:
            continue
        train_dataloaders.append(accelerator.prepare(backend["train_dataloader"]))
    idx_count = 0

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    logging.info("Recalculating max step count.")
    num_update_steps_per_epoch = math.ceil(
        sum([len(dataloader) for dataloader in train_dataloaders])
        / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
        lr_scheduler.num_update_steps_per_epoch = num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(
        "After all of the heave-ho messing around, we have settled on"
        f" {args.num_train_epochs} epochs and {num_update_steps_per_epoch} steps per epoch."
    )

    if vae is not None and not args.keep_vae_loaded and args.vae_cache_preprocess:
        memory_before_unload = torch.cuda.memory_allocated() / 1024**3
        import gc

        del vae
        vae = None
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                backend["vaecache"].vae = None
        reclaim_memory()
        memory_after_unload = torch.cuda.memory_allocated() / 1024**3
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After the VAE from orbit, we freed {abs(round(memory_saved, 2)) * 1024} MB of VRAM."
        )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    global_step = 0
    global_resume_step = 0
    first_epoch = 0
    current_percent_completion = 0
    scheduler_kwargs = {}
    validation = Validation(
        accelerator=accelerator,
        prompt_handler=prompt_handler,
        unet=unet,
        transformer=None,
        args=args,
        validation_prompts=validation_prompts,
        validation_shortnames=validation_shortnames,
        text_encoder_1=text_encoder,
        tokenizer=tokenizer,
        vae_path=args.pretrained_model_name_or_path,
        weight_dtype=weight_dtype,
        embed_cache=StateTracker.get_default_text_embed_cache(),
        validation_negative_pooled_embeds=None,
        validation_negative_prompt_embeds=validation_negative_prompt_embeds,
        text_encoder_2=None,
        tokenizer_2=None,
        ema_unet=ema_unet,
        vae=vae,
        controlnet=controlnet if args.controlnet else None,
    )
    # Potentially load in the weights and states from a previous save
    first_epoch = 1
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logging.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            logging.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            for _, backend in StateTracker.get_data_backends().items():
                if "sampler" in backend:
                    backend["sampler"].load_states(
                        state_path=os.path.join(
                            args.output_dir, path, "training_state.json"
                        ),
                    )
                    backend["sampler"].log_state()
            first_epoch = max(
                [
                    backend["sampler"].current_epoch if "sampler" in backend else 0
                    for _, backend in StateTracker.get_data_backends().items()
                ]
            )
            global_resume_step = global_step = int(path.split("-")[1])
            StateTracker.set_global_resume_step(global_resume_step)
    total_steps_remaining_at_start = args.max_train_steps
    # We store the number of dataset resets that have occurred inside the checkpoint.
    if global_resume_step > 1:
        total_steps_remaining_at_start = (
            total_steps_remaining_at_start - global_resume_step
        )
        logger.debug(
            f"Resuming from epoch {first_epoch}, which leaves us with {total_steps_remaining_at_start}."
        )
    current_epoch = first_epoch
    if current_epoch > args.num_train_epochs + 1:
        logger.info(
            f"Reached the end ({current_epoch} epochs) of our training run ({args.num_train_epochs} epochs). This run will do zero steps."
        )

    # if not use_deepspeed_scheduler:
    #     lr_scheduler = get_lr_scheduler(
    #         args, optimizer, accelerator, logger, use_deepspeed_scheduler=False
    #     )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Remove the args that we don't want to track:
        project_name = args.tracker_project_name or "simpletuner-training"
        tracker_run_name = args.tracker_run_name or "simpletuner-training-run"
        args_hash = hashlib.md5(
            json.dumps(vars(args), sort_keys=True).encode("utf-8")
        ).hexdigest()
        project_name = args.tracker_project_name or "simpletuner-training"
        tracker_run_name = args.tracker_run_name or "simpletuner-training-run"
        accelerator.init_trackers(
            project_name,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": tracker_run_name,
                    "id": f"{args_hash}",
                    "resume": "allow",
                    "allow_val_change": True,
                }
            },
        )

    initial_msg = "\n***** Running training *****"
    total_num_batches = sum(
        [
            len(backend["train_dataset"] if "train_dataset" in backend else [])
            for _, backend in StateTracker.get_data_backends().items()
        ]
    )
    initial_msg += f"\n-  Num batches = {total_num_batches}, unet dtype: `{unet.dtype}`"

    initial_msg += f"\n-  Num Epochs = {args.num_train_epochs}"
    initial_msg += f"\n  - Current Epoch = {first_epoch}"
    initial_msg += f"\n-  Instantaneous batch size per device = {args.train_batch_size}"
    initial_msg += (
        f"\n-  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    initial_msg += f"\n-  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    initial_msg += f"\n-  Total optimization steps = {args.max_train_steps}"
    if global_step > 1:
        initial_msg += f"\n  - Steps completed: {global_step}"
    initial_msg += f"\n-  Total optimization steps remaining = {max(0, total_steps_remaining_at_start)}"
    logger.info(initial_msg)
    if webhook_handler is not None:
        webhook_handler.send(message=initial_msg)
    # Only show the progress bar once on each machine.
    show_progress_bar = True
    if not accelerator.is_local_main_process:
        show_progress_bar = False
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        disable=not show_progress_bar,
        initial=global_step,
        desc=f"Epoch {first_epoch}/{args.num_train_epochs} Steps",
        ncols=125,
    )
    accelerator.wait_for_everyone()

    # Some values that are required to be initialised later.
    timesteps_buffer = []
    train_loss = 0.0
    step = global_step
    training_luminance_values = []
    current_epoch_step = None

    for epoch in range(first_epoch, args.num_train_epochs + 1):
        if current_epoch > args.num_train_epochs + 1:
            # This might immediately end training, but that's useful for simply exporting the model.
            logger.info(
                f"Training run is complete ({args.num_train_epochs}/{args.num_train_epochs} epochs, {global_step}/{args.max_train_steps} steps)."
            )
            break
        if first_epoch != epoch:
            logger.debug(
                f"Just completed epoch {current_epoch}. Beginning epoch {epoch}. Final epoch will be {args.num_train_epochs}"
            )
            for backend_id, backend in StateTracker.get_data_backends().items():
                backend_config = StateTracker.get_data_backend_config(backend_id)
                logger.debug(f"Backend config: {backend_config}")
                if (
                    "crop_aspect" in backend_config
                    and backend_config["crop_aspect"] is not None
                    and backend_config["crop_aspect"] == "random"
                    and "metadata_backend" in backend
                ):
                    # when the aspect ratio is random, we need to shuffle the dataset on each epoch.
                    backend["metadata_backend"].compute_aspect_ratio_bucket_indices(
                        ignore_existing_cache=True
                    )
                    # we have to rebuild the VAE cache if it exists.
                    if "vaecache" in backend:
                        backend["vaecache"].rebuild_cache()
                    backend["metadata_backend"].save_cache()
                elif (
                    "vae_cache_clear_each_epoch" in backend_config
                    and backend_config["vae_cache_clear_each_epoch"]
                    and "vaecache" in backend
                ):
                    # If the user has specified that this should happen,
                    # we will clear the cache and then rebuild it. This is useful for random crops.
                    logger.debug(f"VAE Cache rebuild is enabled. Rebuilding.")
                    logger.debug(f"Backend config: {backend_config}")
                    backend["vaecache"].rebuild_cache()
        current_epoch = epoch
        StateTracker.set_epoch(epoch)
        if args.lr_scheduler == "cosine_with_restarts":
            scheduler_kwargs["epoch"] = epoch

        current_epoch_step = 0
        if args.controlnet:
            controlnet.train()
            training_models = [controlnet]
        else:
            unet.train()
            training_models = [unet]
        if args.train_text_encoder:
            logger.debug(f"Bumping text encoder.")
            text_encoder.train()
            training_models.append(text_encoder)

        if current_epoch_step is not None:
            # We are resetting to the next epoch, if it is not none.
            current_epoch_step = 0
        else:
            # If it's None, we need to calculate the current epoch step based on the current global step.
            current_epoch_step = global_step % num_update_steps_per_epoch
        train_backends = {}
        for backend_id, backend in StateTracker.get_data_backends().items():
            if StateTracker.backend_status(backend_id):
                # Exclude exhausted backends.
                continue
            if "train_dataloader" in backend:
                train_backends[backend_id] = backend["train_dataloader"]

        for step, batch in random_dataloader_iterator(train_backends):
            if args.lr_scheduler == "cosine_with_restarts":
                scheduler_kwargs["step"] = global_step

            if accelerator.is_main_process:
                progress_bar.set_description(
                    f"Epoch {current_epoch}/{args.num_train_epochs}, Steps"
                )

            # If we receive a False from the enumerator, we know we reached the next epoch.
            if batch is False:
                logger.info(f"Reached the end of epoch {epoch}")
                break

            if batch is None:
                import traceback

                raise ValueError(
                    f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                )

            if "batch_luminance" in batch:
                # Add the current batch of training data's avg luminance to a list.
                training_luminance_values.append(batch["batch_luminance"])

            with accelerator.accumulate(training_models):
                training_logger.debug(f"Sending latent batch to GPU")
                latents = batch["latent_batch"].to(
                    accelerator.device, dtype=weight_dtype
                )

                # Sample noise that we'll add to the latents - args.noise_offset might need to be set to 0.1 by default.
                if args.offset_noise:
                    if (
                        args.noise_offset_probability == 1.0
                        or random.random() < args.noise_offset_probability
                    ):
                        noise = torch.randn_like(
                            latents
                        ) + args.noise_offset * torch.randn(
                            latents.shape[0],
                            latents.shape[1],
                            1,
                            1,
                            device=latents.device,
                        )
                else:
                    noise = torch.randn_like(latents)
                bsz, channels, height, width = latents.shape

                logger.debug(f"Working on batch size: {bsz}")
                # Sample a random timestep for each image, potentially biased by the timestep weights.
                # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                weights = generate_timestep_weights(
                    args, noise_scheduler.config.num_train_timesteps
                ).to(accelerator.device)
                # Instead of uniformly sampling the timestep range, we'll split our weights and schedule into bsz number of segments.
                # This enables more broad sampling and potentially more effective training.
                if bsz > 1 and not args.disable_segmented_timestep_sampling:
                    timesteps = segmented_timestep_selection(
                        actual_num_timesteps=noise_scheduler.config.num_train_timesteps,
                        bsz=bsz,
                        weights=weights,
                    ).to(accelerator.device)
                else:
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Prepare the data for the scatter plot
                for timestep in timesteps.tolist():
                    timesteps_buffer.append((global_step, timestep))

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(
                    accelerator.device
                )

                # Get the text embedding for conditioning
                encoder_hidden_states = batch["prompt_embeds"].to(
                    device=accelerator.device,
                    dtype=(
                        text_encoder.dtype if text_encoder is not None else weight_dtype
                    ),
                )
                training_logger.debug(
                    f"Encoder hidden states: {encoder_hidden_states.shape}"
                )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    target = latents
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        "Supported types are 'epsilon', `sample`, and 'v_prediction'."
                    )

                # Predict the noise residual
                training_logger.debug(f"Running prediction")
                training_logger.debug(
                    f"\n -> Latents device: {latents.device}"
                    f"\n -> Noise device: {noise.device}"
                    f"\n -> Timesteps device: {timesteps.device}"
                    f"\n -> Encoder hidden states device: {encoder_hidden_states.device}"
                    f"\n -> Latents dtype: {latents.dtype}, shape: {latents.shape if hasattr(latents, 'shape') else 'None'}"
                    f"\n -> Noise dtype: {noise.dtype}, shape: {noise.shape if hasattr(noise, 'shape') else 'None'}"
                    f"\n -> Timesteps dtype: {timesteps.dtype}"
                    f"\n -> Encoder hidden states dtype: {encoder_hidden_states.dtype}, shape: {encoder_hidden_states.shape if hasattr(encoder_hidden_states, 'shape') else 'None'}"
                )
                if unwrap_model(accelerator, unet).config.in_channels == channels * 2:
                    # deepfloyd stage ii requires the inputs to be doubled. note that we're working in pixels, not latents.
                    noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=1)

                if "deepfloyd-stage2" in args.model_type:
                    class_labels = timesteps
                else:
                    class_labels = None
                if args.controlnet:
                    # ControlNet conditioning.
                    controlnet_image = batch["conditioning_pixel_values"].to(
                        dtype=weight_dtype
                    )
                    training_logger.debug(f"Image shape: {controlnet_image.shape}")
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )
                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype)
                            for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(
                            dtype=weight_dtype
                        ),
                        return_dict=False,
                    )[0]
                else:
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        class_labels=class_labels,
                    ).sample

                if model_pred.shape[1] == 6:
                    # Chop the variance off of DeepFloyd models.
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # x-prediction requires that we now subtract the noise residual from the prediction to get the target sample.
                if noise_scheduler.config.prediction_type == "sample":
                    model_pred = model_pred - noise

                if args.snr_gamma is None:
                    training_logger.debug(
                        f"Calculating loss for {model_pred.shape} vs {target.shape}"
                    )
                    loss = args.snr_weight * F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    training_logger.debug(f"Using min-SNR loss")
                    snr = compute_snr(
                        timesteps=timesteps,
                        noise_scheduler=noise_scheduler,
                        use_soft_min=(
                            True
                            if "deepfloyd" in args.model_type
                            or args.use_soft_min_snr is True
                            else False
                        ),
                        sigma_data=(
                            1.0
                            if args.soft_min_snr_sigma_data is None
                            else args.soft_min_snr_sigma_data
                        ),
                    )
                    snr_divisor = snr
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr_divisor = snr + 1

                    training_logger.debug(
                        f"Calculating MSE loss weights using SNR as divisor"
                    )
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr_divisor
                    )

                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    training_logger.debug(
                        f"Calculating original MSE loss without reduction"
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    training_logger.debug(f"Calculating SNR-weighted MSE loss")
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    training_logger.debug(f"Reducing loss via mean")
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                logger.debug(f"Backwards pass.")
                accelerator.backward(loss)
                grad_norm = None
                if (
                    accelerator.sync_gradients
                    and not args.use_adafactor_optimizer
                    and args.max_grad_norm > 0
                ):
                    # Adafactor shouldn't have gradient clipping applied.
                    grad_norm = accelerator.clip_grad_norm_(
                        params_to_optimize, args.max_grad_norm
                    )
                training_logger.debug(f"Stepping components forward.")
                optimizer.step()
                lr_scheduler.step(**scheduler_kwargs)
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                try:
                    if args.use_adafactor_optimizer:
                        lr = lr_scheduler.get_lr()[0]
                    else:
                        lr = lr_scheduler.get_last_lr()[0]
                except Exception as e:
                    logger.error(
                        f"Failed to get the last learning rate from the scheduler. Error: {e}"
                    )
                logs = {
                    "train_loss": train_loss,
                    "optimization_loss": loss,
                    "learning_rate": lr,
                    "epoch": epoch,
                }
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm
                progress_bar.update(1)
                global_step += 1
                current_epoch_step += 1
                StateTracker.set_global_step(global_step)

                # SD 2.x uses the current_percent_completion to handle finishing the text encoder training early.
                current_percent_completion = int(
                    progress_bar.n / progress_bar.total * 100
                )

                ema_decay_value = "None (EMA not in use)"
                if args.use_ema:
                    training_logger.debug(f"Stepping EMA unet forward")
                    ema_unet.step(unet.parameters())
                    # There seems to be an issue with EMAmodel not keeping proper track of itself.
                    ema_unet.optimization_step = global_step
                    ema_decay_value = ema_unet.get_decay(ema_unet.optimization_step)
                    logs["ema_decay_value"] = ema_decay_value

                # Log scatter plot to wandb
                if args.report_to == "wandb" and accelerator.is_main_process:
                    # Prepare the data for the scatter plot
                    data = [
                        [iteration, timestep]
                        for iteration, timestep in timesteps_buffer
                    ]
                    table = wandb.Table(data=data, columns=["global_step", "timestep"])
                    logs["timesteps_scatter"] = wandb.plot.scatter(
                        table,
                        "global_step",
                        "timestep",
                        title="Timestep distribution by step",
                    )

                # Clear buffers
                timesteps_buffer = []

                # Average out the luminance values of each batch, so that we can store that in this step.
                avg_training_data_luminance = sum(training_luminance_values) / len(
                    training_luminance_values
                )
                logs["train_luminance"] = avg_training_data_luminance

                logger.debug(
                    f"Step {global_step} of {args.max_train_steps}: loss {loss.item()}, lr {lr}, epoch {epoch}/{args.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {train_loss}"
                )
                accelerator.log(
                    logs,
                    step=global_step,
                )
                if webhook_handler is not None:
                    webhook_pending_msg = f"Step {global_step} of {args.max_train_steps}: loss {round(loss.item(), 4)}, lr {lr}, epoch {epoch}/{args.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {round(train_loss, 4)}"
                # Reset some values for the next go.
                training_luminance_values = []
                train_loss = 0.0

                if (
                    args.freeze_encoder
                    and args.train_text_encoder
                    and text_encoder is not None
                    and current_percent_completion > args.text_encoder_limit
                ):
                    # We want to stop training the text_encoder around 25% by default.
                    freeze_entire_component(text_encoder)
                    logger.warning(
                        f"Frozen text_encoder at {current_percent_completion}%!"
                    )
                    # This will help ensure we don't run this check every time from now on.
                    args.freeze_encoder = False
                    args.train_text_encoder = False

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if webhook_handler is not None:
                            webhook_handler.send(
                                message=f"Checkpoint: `{webhook_pending_msg}`",
                                message_level="info",
                            )
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.debug(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.debug(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process or use_deepspeed_optimizer:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        print("\n")
                        accelerator.save_state(save_path)
                        for _, backend in StateTracker.get_data_backends().items():
                            if "sampler" in backend:
                                logger.debug(f"Saving backend state: {backend}")
                                backend["sampler"].save_state(
                                    state_path=os.path.join(
                                        save_path, "training_state.json"
                                    ),
                                )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            validation.run_validations(validation_type="intermediary", step=step)
            if (
                args.push_to_hub
                and args.push_checkpoints_to_hub
                and global_step % args.checkpointing_steps == 0
                and step % args.gradient_accumulation_steps == 0
                and global_step > global_resume_step
            ):
                if accelerator.is_main_process:
                    try:
                        hub_manager.upload_latest_checkpoint(
                            validation_images=validation.validation_images,
                            webhook_handler=webhook_handler,
                        )
                    except Exception as e:
                        logger.error(f"Failed to push checkpoint to hub: {e}")
            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps or epoch > args.num_train_epochs + 1:
                logger.info(
                    f"Training run is complete ({args.num_train_epochs}/{args.num_train_epochs} epochs, {global_step}/{args.max_train_steps} steps)."
                )
                break

        if global_step >= args.max_train_steps or epoch > args.num_train_epochs + 1:
            logger.info(
                f"Exiting training loop. Beginning model unwind at epoch {epoch}, step {global_step}"
            )
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        validation_images = validation.run_validations(
            validation_type="final",
            step=global_step,
            force_evaluation=True,
            skip_execution=True,
        ).validation_images
        unet = accelerator.unwrap_model(unet)
        if args.model_type == "full" and args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
        elif "lora" in args.model_type:
            unet_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet)
            )
            if args.train_text_encoder:
                text_encoder = accelerator.unwrap_model(text_encoder)
                text_encoder_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(text_encoder)
                )
            else:
                text_encoder_lora_layers = None

            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
            )

            del text_encoder_lora_layers
            reclaim_memory()

        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        if StateTracker.get_vae() is None and "deepfloyd" not in args.model_type:
            if webhook_handler is not None:
                webhook_handler.send(message="Loading VAE..")
            StateTracker.set_vae(
                AutoencoderKL.from_pretrained(
                    args.pretrained_vae_model_name_or_path,
                    subfolder=(
                        "vae"
                        if args.pretrained_vae_model_name_or_path is None
                        else None
                    ),
                    revision=args.revision,
                    variant=args.variant,
                    force_upcast=False,
                )
            )
        if "deepfloyd" in args.model_type:
            from diffusers import DiffusionPipeline

            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                unet=unet,
                revision=args.revision,
                variant=args.variant,
                safety_checker=None,
                watermarker=None,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=StateTracker.get_vae(),
                unet=unet,
                revision=args.revision,
                variant=args.variant,
                safety_checker=None,
            )
            pipeline.scheduler = SCHEDULER_NAME_MAP[
                args.validation_noise_scheduler
            ].from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="scheduler",
                prediction_type=args.prediction_type,
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True,
                revision=args.revision,
            )
        if "full" in args.model_type:
            pipeline.save_pretrained(
                os.path.join(args.output_dir, args.hub_model_id or "pipeline"),
                safe_serialization=True,
            )

        if args.push_to_hub:
            hub_manager.upload_model(validation_images, webhook_handler)

        else:
            if webhook_handler is not None:
                webhook_handler.send(
                    message=f"\n# Training has completed after {current_epoch} epochs and {global_step} steps."
                )

    accelerator.end_training()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("fork")
    try:
        main()
    except KeyboardInterrupt as e:
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message="Training has been interrupted by user action (lost terminal, or ctrl+C)."
            )
    except Exception as e:
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Training has failed. Please check the logs for more information: {e}"
            )
        import traceback

        logger.error(f"Epic fail: {e}, traceback: {traceback.format_exc()}")
