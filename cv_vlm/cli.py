from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from cv_vlm.navigation.goal import compute_goal
from cv_vlm.pipeline.offline_builder import BuilderConfig, Intrinsics, OfflineMapBuilder
from cv_vlm.mapping.object_store import ObjectStore

app = typer.Typer(help="Semantic navigation toolkit for TUM RGB-D")


def _load_store(map_dir: Path) -> ObjectStore:
    return ObjectStore.load(map_dir)


@app.command("build-map")
def build_map(
    dataset_dir: Path = typer.Argument(..., help="Path to TUM RGB-D sequence directory"),
    output_dir: Path = typer.Argument(..., help="Where to store the generated map"),
    fx: float = typer.Option(525.0, help="Camera focal length fx"),
    fy: float = typer.Option(525.0, help="Camera focal length fy"),
    cx: float = typer.Option(319.5, help="Camera principal point cx"),
    cy: float = typer.Option(239.5, help="Camera principal point cy"),
    depth_scale: float = typer.Option(5000.0, help="Depth scale to convert raw units to meters"),
    default_class: str = typer.Option("object", help="Default class name assigned to masks"),
    merge_distance: float = typer.Option(0.6, help="Merge distance in meters for object clustering"),
    near_percentile: float = typer.Option(10.0, help="Lower depth percentile used for mask extraction"),
    far_percentile: float = typer.Option(70.0, help="Upper depth percentile used for mask extraction"),
    min_area: int = typer.Option(300, help="Minimum pixel area for a mask"),
    max_points_per_mask: int = typer.Option(8000, help="Maximum sampled points per mask"),
):
    intr = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    config = BuilderConfig(
        dataset_root=dataset_dir,
        output_dir=output_dir,
        intrinsics=intr,
        depth_scale=depth_scale,
        default_class=default_class,
        merge_distance=merge_distance,
        near_percentile=near_percentile,
        far_percentile=far_percentile,
        min_area=min_area,
        max_points_per_mask=max_points_per_mask,
    )
    builder = OfflineMapBuilder(config)
    store = builder.run()
    typer.echo(f"Saved {len(store.objects())} objects to {output_dir}")


@app.command()
def query(map_dir: Path = typer.Argument(..., help="Directory with stored map"), class_name: str = typer.Argument(..., help="Class name to search")):
    store = _load_store(map_dir)
    matches = store.query(class_name)
    if not matches:
        typer.echo("No objects found")
        raise typer.Exit(code=1)
    for obj in matches:
        typer.echo(
            f"id={obj.object_id} class={obj.class_name} centroid={obj.centroid.round(3)} observations={obj.observations} points={obj.points_count} last_seen={obj.last_seen:.2f}"
        )


@app.command()
def goto(map_dir: Path = typer.Argument(..., help="Directory with stored map"), object_id: int = typer.Argument(..., help="Object id"), offset: float = typer.Option(0.8, help="Offset distance in meters")):
    store = _load_store(map_dir)
    obj = next((o for o in store.objects() if o.object_id == object_id), None)
    if obj is None:
        typer.echo(f"Object {object_id} not found")
        raise typer.Exit(code=1)
    goal = compute_goal(obj, offset_m=offset)
    typer.echo(f"Goal: {goal.round(3)} (offset {offset} m from object centroid {obj.centroid.round(3)})")


if __name__ == "__main__":
    app()
