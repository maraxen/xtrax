from dataclasses import dataclass
from typing import Any

from xtrax.cli.loader import CLIImportError, load_fn
from xtrax.data.module import DataModule


@dataclass
class ResolvedComponents:
    model: Any
    optimizer: Any
    loss_fn: Any
    dataset: DataModule


def _resolve(section: str, path: str, kwargs: dict):
    """Wrap load_fn so CLIImportError names the section."""
    try:
        return load_fn(path)(**kwargs)
    except CLIImportError as e:
        raise CLIImportError(f"[{section}] {e}") from e


def resolve_components(manifest_dict: dict, epochs: int) -> ResolvedComponents:
    """Resolve components from a manifest dictionary."""
    model = _resolve(
        "model",
        manifest_dict["model"]["path"],
        manifest_dict["model"].get("kwargs", {}),
    )
    optimizer = _resolve(
        "optimizer",
        manifest_dict["optimizer"]["path"],
        manifest_dict["optimizer"].get("kwargs", {}),
    )
    loss_fn = _resolve(
        "loss",
        manifest_dict["loss"]["path"],
        manifest_dict["loss"].get("kwargs", {}),
    )

    dataset = _resolve(
        "data",
        manifest_dict["data"]["factory"],
        manifest_dict["data"].get("kwargs", {}),
    )

    data = DataModule(
        dataset,
        batch_size=manifest_dict["data"]["batch_size"],
        num_epochs=epochs,
        seed=manifest_dict["seed"],
        distributed=False,
    )

    return ResolvedComponents(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataset=data,
    )
