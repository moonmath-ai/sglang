import copy
import logging
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.gguf_load import load_gguf_transformer
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_gguf_transformer_path,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import _normalize_component_type
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.srt.utils import is_npu

_is_npu = is_npu()

logger = init_logger(__name__)


def _server_args_for_transformer_component(
    server_args: ServerArgs, component_name: str
) -> ServerArgs:
    """Mask global quantized override flags for secondary transformer components."""
    if component_name not in ("transformer_2", "unconditional_transformer"):
        return server_args

    # Some pipelines have secondary DiT components with their own quantized
    # weight file. Keep the mapping model-owned and the loader generic.
    component_weights_paths = getattr(
        server_args, "component_transformer_weights_paths", {}
    )
    component_weights_path = component_weights_paths.get(component_name)
    if component_weights_path is not None:
        component_server_args = copy.copy(server_args)
        component_server_args.transformer_weights_path = component_weights_path
        component_server_args.nunchaku_config = None
        logger.info(
            "Using transformer_weights_path override for %s: %s",
            component_name,
            component_weights_path,
        )
        return component_server_args

    if (
        server_args.transformer_weights_path is None
        and server_args.nunchaku_config is None
    ):
        return server_args

    component_server_args = copy.copy(server_args)
    component_server_args.transformer_weights_path = None
    component_server_args.nunchaku_config = None
    logger.info(
        "Ignoring global transformer_weights_path for %s; keep it on the base "
        "checkpoint unless a per-component override path is provided.",
        component_name,
    )
    return component_server_args


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = [
        "transformer",
        "unconditional_transformer",
        "audio_dit",
        "video_dit",
    ]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        component_server_args = _server_args_for_transformer_component(
            server_args, component_name
        )

        # 1. hf config
        config = get_diffusers_component_config(component_path=component_model_path)

        # GGUF transformers ship as a single .gguf file and load via a dedicated
        # on-the-fly-dequant path; the base model still provides the arch config.
        gguf_path = resolve_gguf_transformer_path(component_server_args)

        if gguf_path is None:
            safetensors_list = resolve_transformer_safetensors_to_load(
                component_server_args, component_model_path
            )
        else:
            safetensors_list = []

        # 2. dit config
        # Config from Diffusers supersedes sgl_diffusion's model config
        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path
        if component_name in ("transformer", "unconditional_transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        cls_name = config.pop("_class_name")
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        if gguf_path is not None:
            return self._load_gguf(
                model_cls=model_cls,
                cls_name=cls_name,
                config=config,
                dit_config=dit_config,
                gguf_path=gguf_path,
                server_args=server_args,
            )

        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=config,
            server_args=component_server_args,
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            model_cls=model_cls,
            cls_name=cls_name,
        )

        logger.info(
            "Loading %s from %s safetensors file(s) %s, param_dtype: %s",
            cls_name,
            len(safetensors_list),
            f": {safetensors_list}" if get_log_level() == logging.DEBUG else "",
            quant_spec.param_dtype,
        )
        # prepare init_param
        init_params: dict[str, Any] = {
            "config": dit_config,
            "hf_config": config,
            "quant_config": quant_spec.runtime_quant_config,
        }
        if (
            init_params["quant_config"] is None
            and component_server_args.transformer_weights_path is not None
        ):
            logger.warning(
                "transformer_weights_path provided, but quantization config not resolved, which is unexpected and likely to cause errors"
            )
        else:
            logger.debug("quantization config: %s", init_params["quant_config"])

        # Load the model using FSDP loader
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params=init_params,
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=component_server_args.dit_cpu_offload,
            pin_cpu_memory=component_server_args.pin_cpu_memory,
            fsdp_inference=component_server_args.use_fsdp_inference,
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        # post-hooks (e.g., patch scales (nunchaku))
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(model)

        # considering the existent of mixed-precision models (e.g., nunchaku)
        if (
            next(model.parameters()).dtype != quant_spec.param_dtype
            and quant_spec.param_dtype
        ):
            logger.warning(
                "Model dtype does not match expected param dtype, %s vs %s",
                next(model.parameters()).dtype,
                quant_spec.param_dtype,
            )

        return model

    def _load_gguf(
        self,
        *,
        model_cls: type,
        cls_name: str,
        config: dict[str, Any],
        dit_config: Any,
        gguf_path: str,
        server_args: ServerArgs,
    ):
        """Load a GGUF-quantized transformer (on-the-fly dequant, single GPU)."""
        from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig
        from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

        if server_args.tp_size > 1 or server_args.use_fsdp_inference:
            raise ValueError(
                "GGUF transformer loading does not support tensor parallelism or "
                "FSDP inference; run with tp_size=1 and FSDP inference disabled."
            )

        quant_config = GGUFConfig()
        packed = getattr(model_cls, "packed_modules_mapping", None)
        if packed:
            quant_config.packed_modules_mapping = packed

        param_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        init_params: dict[str, Any] = {
            "config": dit_config,
            "hf_config": config,
            "quant_config": quant_config,
        }

        logger.info("Loading GGUF %s from %s", cls_name, gguf_path)
        model = load_gguf_transformer(
            model_cls=model_cls,
            init_params=init_params,
            gguf_file=gguf_path,
            device=get_local_torch_device(),
            param_dtype=param_dtype,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded GGUF model with %.2fB parameters", total_params / 1e9)
        return model
