"""Create a minimal container build context without workspace secrets or UI."""

import argparse
import shutil
from pathlib import Path


def build_context(target, assets=None):
    root = Path(__file__).resolve().parents[2]
    target = Path(target).resolve()
    if assets and not Path(assets).expanduser().is_dir():
        raise ValueError("router asset directory is unavailable")
    if target.exists():
        raise ValueError("build context must be a new directory")
    target.mkdir(parents=True)
    shutil.copy2(root / "README.md", target / "README.md")
    shutil.copy2(root / "infra/gateway-gcp/gateway.pyproject.toml", target / "pyproject.toml")
    shutil.copytree(
        root / "theriad", target / "theriad", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(root / "infra/gateway-gcp/Dockerfile", target / "Dockerfile")
    destination = target / "router-assets"
    if assets:
        shutil.copytree(Path(assets).expanduser(), destination)
    else:
        destination.mkdir()
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("--router-assets")
    args = parser.parse_args()
    print(build_context(args.target, args.router_assets))
