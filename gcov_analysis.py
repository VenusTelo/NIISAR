"""Compara ventanas pequeñas de dos observaciones NISAR GCOV validadas."""

import base64
import io
import math
import os
from datetime import date, timedelta
from urllib.parse import urlparse

import asf_search as asf
import fsspec
import h5py
import matplotlib
import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape

import parameters


GCOV_GROUP = "/science/LSAR/GCOV/grids/frequencyA"
DOWNSAMPLE = 4
MAX_AREA_PIXELS = 400_000


class AnalysisError(Exception):
    """Condición esperada que impide realizar el análisis."""


def validate_area(coordinates):
    if not isinstance(coordinates, list) or not 4 <= len(coordinates) <= 501:
        raise AnalysisError("Draw a closed polygon with between 3 and 500 vertices.")
    points = []
    for point in coordinates:
        if (not isinstance(point, list) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       for v in point)):
            raise AnalysisError("The polygon contains invalid coordinates.")
        lon, lat = point
        if not (-180 <= lon <= 180 and -90 <= lat <= 90
                and math.isfinite(lon) and math.isfinite(lat)):
            raise AnalysisError("The polygon contains coordinates outside the valid range.")
        points.append((lon, lat))
    if points[0] != points[-1]:
        raise AnalysisError("The polygon must be closed.")
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid or polygon.area == 0:
        raise AnalysisError("The polygon must have a valid area and must not self-intersect.")
    return polygon


def validate_dates(mode, start, end):
    if mode not in ("single", "compare"):
        raise AnalysisError("Select a date or a comparison range.")
    try:
        first = date.fromisoformat(start)
        last = date.fromisoformat(end) if mode == "compare" else first
    except (TypeError, ValueError):
        raise AnalysisError("Enter valid dates in YYYY-MM-DD format.") from None
    if first > last:
        raise AnalysisError("The start date must be on or before the end date.")
    if mode == "compare" and first == last:
        raise AnalysisError("To compare, select a range of at least two days.")
    if last == date.max:
        raise AnalysisError("The end date is outside the allowed range.")
    return first, last


def fetch_area_observations(polygon, first=None, last=None):
    options = {"shortName": "NISAR_L2_GCOV_V1", "intersectsWith": polygon.wkt}
    if first is not None and last is not None:
        options["start"] = f"{first.isoformat()}T00:00:00Z"
        options["end"] = f"{(last + timedelta(days=1)).isoformat()}T00:00:00Z"
    results = asf.search(**options)
    groups = {}
    for result in results:
        props = dict(result.properties)
        attrs = {
            entry["Name"]: entry.get("Values", [])
            for entry in getattr(result, "umm", {}).get("AdditionalAttributes", [])
            if "Name" in entry
        }
        geometry = getattr(result, "geometry", None)
        if not geometry or not shape(geometry).covers(polygon):
            continue
        polarizations = attrs.get("FREQUENCY_A_POLARIZATION")
        if polarizations and "HV" not in polarizations:
            continue
        try:
            h5_url(props)
        except AnalysisError:
            continue
        path = props.get("pathNumber")
        frame = props.get("frameNumber")
        if path is None and attrs.get("TRACK_NUMBER"):
            path = int(attrs["TRACK_NUMBER"][0])
        if frame is None and attrs.get("FRAME_NUMBER"):
            frame = int(attrs["FRAME_NUMBER"][0])
        direction = props.get("flightDirection")
        day = str(props.get("startTime") or "")[:10]
        if path is None or frame is None or not direction:
            continue
        try:
            observed = date.fromisoformat(day)
        except ValueError:
            continue
        if first is not None and last is not None and not first <= observed <= last:
            continue
        props["pathNumber"], props["frameNumber"] = path, frame
        props["_full_frame"] = attrs.get("FULL_FRAME", ["FALSE"])[0] == "TRUE"
        dates = groups.setdefault((path, frame, direction), {})
        if day not in dates or props["_full_frame"] > dates[day]["_full_frame"]:
            dates[day] = props
    return groups


def available_area_dates(groups):
    all_dates = sorted({day for dates in groups.values() for day in dates})
    compatible = {day: set() for day in all_dates}
    for dates in groups.values():
        ordered = sorted(dates)
        for index, day in enumerate(ordered):
            compatible[day].update(ordered[index + 1:])
    return {"dates": all_dates, "compatible": {day: sorted(days) for day, days in compatible.items()}}


def select_area_observations(polygon, first, last, mode, groups=None):
    if groups is None:
        groups = fetch_area_observations(polygon, first, last)
    else:
        groups = {
            key: {day: item for day, item in dates.items()
                  if first.isoformat() <= day <= last.isoformat()}
            for key, dates in groups.items()
        }

    all_dates = sorted({day for dates in groups.values() for day in dates})
    if not all_dates:
        raise AnalysisError(
            "No validated GCOV scene fully covering the area was found for the selected dates."
        )
    if mode == "single":
        choices = [dates[all_dates[0]] for dates in groups.values() if all_dates[0] in dates]
        return [max(choices, key=lambda item: item["_full_frame"])]
    if len(all_dates) < 2:
        raise AnalysisError("At least two distinct GCOV dates are needed within the selected range.")
    earliest, latest = all_dates[0], all_dates[-1]
    choices = [(dates[earliest], dates[latest]) for dates in groups.values()
               if earliest in dates and latest in dates]
    if not choices:
        raise AnalysisError(
            "The first and last available dates do not share the same path, frame and direction "
            "with full coverage. Select another range or a smaller area."
        )
    return list(max(choices, key=lambda pair: pair[0]["_full_frame"] and pair[1]["_full_frame"]))


def h5_url(properties):
    urls = [properties.get("url")] + list(properties.get("additionalUrls") or [])
    for url in urls:
        if not isinstance(url, str):
            continue
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        trusted = host.endswith(".nasa.gov") or host.endswith(".asf.alaska.edu")
        if (parsed.scheme == "https" and trusted and parsed.path.lower().endswith(".h5")
                and "QA_STATS" not in parsed.path.upper()):
            return url
    raise AnalysisError("The observation does not provide a compatible GCOV HDF5 file from NASA or ASF.")


def _read_layers(fs, url, bbox, polygon=None, names=("HVHV",), optional=()):
    """Lee varias capas de un GCOV con UNA sola apertura del archivo remoto.

    Devuelve (capas, epsg, transform, area_mask, factor, método). `capas` es un
    dict nombre -> array (potencia lineal; NaN fuera del área o sin dato).
    Las capas en `optional` se omiten si el archivo no las trae.
    """
    west, south, east, north = bbox
    with fs.open(url, "rb", block_size=16 * 1024 * 1024, cache_type="background") as source:
        with h5py.File(source, "r") as hdf:
            group = hdf[GCOV_GROUP]
            epsg = int(group["projection"][()])
            x = group["xCoordinates"][:]
            y = group["yCoordinates"][:]
            dx = float(group["xCoordinateSpacing"][()])
            dy = float(group["yCoordinateSpacing"][()])
            if dx <= 0 or dy >= 0:
                raise AnalysisError("The GCOV grid orientation is not supported.")
            transformer = Transformer.from_crs(4326, epsg, always_xy=True)
            projected = shape(transform_geom("EPSG:4326", f"EPSG:{epsg}", mapping(polygon))) if polygon is not None else None
            if projected is None:
                xs, ys = transformer.transform(
                    [west, east, west, east], [south, south, north, north]
                )
                bounds = (min(xs), min(ys), max(xs), max(ys))
            else:
                bounds = projected.bounds
                grid_bounds = (x[0] - dx / 2, y[-1] + dy / 2, x[-1] + dx / 2, y[0] - dy / 2)
                if (bounds[0] < grid_bounds[0] or bounds[1] < grid_bounds[1]
                        or bounds[2] > grid_bounds[2] or bounds[3] > grid_bounds[3]):
                    raise AnalysisError("The file's grid does not cover the entire selected area.")
            cols = np.flatnonzero((x >= bounds[0]) & (x <= bounds[2]))
            rows = np.flatnonzero((y >= bounds[1]) & (y <= bounds[3]))
            if not len(cols) or not len(rows):
                raise AnalysisError("The selected window is outside the file's coverage.")
            factor = DOWNSAMPLE
            while True:
                i0 = (int(cols.min()) // factor) * factor
                j0 = (int(rows.min()) // factor) * factor
                nx = ((int(cols.max()) + 1 - i0 + factor - 1) // factor) * factor
                ny = ((int(rows.max()) + 1 - j0 + factor - 1) // factor) * factor
                if polygon is None or (nx // factor) * (ny // factor) <= MAX_AREA_PIXELS:
                    break
                factor *= 2
            if nx == 0 or ny == 0:
                raise AnalysisError("The available window is too small to analyze.")
            present = [n for n in names if n in group or n not in optional]

            if factor > DOWNSAMPLE:
                transform = Affine(dx * factor, 0, x[i0] - dx * factor / 2,
                                   0, dy * factor, y[j0] - dy * factor / 2)
                layers, area_mask = {}, None
                for name in present:
                    data = group[name][j0:j0 + ny:factor, i0:i0 + nx:factor].astype("float32")
                    if area_mask is None:
                        area_mask = geometry_mask(
                            [mapping(projected)], out_shape=data.shape, transform=transform, invert=True,
                        )
                    data[~np.isfinite(data) | (data <= 0) | ~area_mask] = np.nan
                    layers[name] = data
                return layers, epsg, transform, area_mask, factor, "muestreo"

            shape_out = (ny // factor, nx // factor)
            layers = {name: np.full(shape_out, np.nan, dtype="float32") for name in present}
            area_mask = np.zeros(shape_out, dtype=bool)
            for offset in range(0, ny, 1024):
                height = min(1024, ny - offset)
                rows_out = slice(offset // factor, (offset + height) // factor)
                if projected is None:
                    inside = np.ones((height, nx), dtype=bool)
                else:
                    inside = geometry_mask(
                        [mapping(projected)], out_shape=(height, nx),
                        transform=Affine(dx, 0, x[i0] - dx / 2,
                                         0, dy, y[j0] - dy / 2 + dy * offset), invert=True,
                    )
                area_mask[rows_out] = inside.reshape(
                    height // factor, factor, nx // factor, factor).any(axis=(1, 3))
                for name in present:
                    raw = group[name][j0 + offset:j0 + offset + height, i0:i0 + nx].astype("float32")
                    data = np.full((height, nx), np.nan, dtype="float32")
                    data[:raw.shape[0], :raw.shape[1]] = raw
                    valid = np.isfinite(data) & (data > 0) & inside
                    shape_blocks = (height // factor, factor, nx // factor, factor)
                    counts = valid.reshape(shape_blocks).sum(axis=(1, 3))
                    sums = np.where(valid, data, 0).reshape(shape_blocks).sum(axis=(1, 3))
                    np.divide(sums, counts, out=layers[name][rows_out], where=counts > 0)
            transform = Affine(dx * factor, 0, x[i0] - dx / 2,
                               0, dy * factor, y[j0] - dy / 2)
            return layers, epsg, transform, area_mask, factor, "promedio"


def image_data(data, low, high, palette):
    valid = np.isfinite(data)
    scaled = np.clip(np.where(valid, (data - low) / (high - low), 0), 0, 1)
    rgba = matplotlib.colormaps[palette](scaled, bytes=True)
    rgba[~valid] = (0, 0, 0, 0)
    image = Image.fromarray(rgba, "RGBA")
    image.thumbnail((600, 600), Image.Resampling.NEAREST)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _png_url(image):
    image.thumbnail((600, 600), Image.Resampling.NEAREST)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def category_payload(data):
    """Convierte una categoría (arrays) en imágenes y datos listos para JSON."""
    rgba = parameters.classes_to_rgba(data["classes"], data["confidence"], data["legend"])
    technical = []
    for title, array, low, high, palette in data["technical"]:
        if high is None:
            finite = array[np.isfinite(array)]
            high = float(np.percentile(finite, 98)) if finite.size else 1.0
            high = high if high > low else low + 1.0
        technical.append({"title": title, "range": [low, high],
                          "image": image_data(array, low, high, palette)})
    return {
        "image": _png_url(Image.fromarray(rgba, "RGBA")),
        "legend": [{"code": code, "label": label, "color": color, "is_change": is_change}
                   for code, (label, color, is_change) in data["legend"].items()],
        "stats": data["stats"],
        "variables": data.get("variables", []),
        "summary": data["summary"],
        "technical": technical,
    }


def analyze_area(coordinates, mode, start, end, groups=None):
    polygon = validate_area(coordinates)
    first_date, last_date = validate_dates(mode, start, end)
    token = os.environ.get("EDL_TOKEN", "").strip()
    if not token or token == "your_earthdata_token_here":
        raise AnalysisError("Set a valid Earthdata token in EDL_TOKEN inside the .env file.")
    observations = select_area_observations(polygon, first_date, last_date, mode, groups)
    fs = fsspec.filesystem("https", headers={"Authorization": f"Bearer {token}"})
    windows = [
        _read_layers(fs, h5_url(item), polygon.bounds, polygon,
                     names=("HVHV", "HHHH", "numberOfLooks"), optional=("HHHH", "numberOfLooks"))
        for item in observations
    ]
    layers_a, epsg, transform, area_mask, factor, sampling = windows[0]
    first = layers_a["HVHV"]
    if not area_mask.any():
        raise AnalysisError("The selected area contains no GCOV pixel centers.")
    with np.errstate(divide="ignore", invalid="ignore"):
        first_db = 10 * np.log10(first)
    result = {
        "mode": mode,
        "dates": [str(item["startTime"])[:10] for item in observations],
        "path": observations[0]["pathNumber"],
        "frame": observations[0]["frameNumber"],
        "direction": observations[0]["flightDirection"],
        "pixel_factor": factor,
        "sampling": sampling,
    }
    pixel_area_km2 = None if epsg == 4326 else abs(transform.a * transform.e) / 1e6
    gap_days = None
    layers_b = None

    if mode == "single":
        valid = np.isfinite(first_db) & area_mask
        if not valid.any():
            raise AnalysisError("There are no valid pixels inside the selected area.")
        result.update({
            "valid_percent": float(valid.sum() / area_mask.sum() * 100),
            "median_hv_db": float(np.median(first_db[valid])),
            "images": {"single": image_data(first_db, -30, -10, "gray")},
        })
    else:
        layers_b, last_epsg, last_transform, last_mask, last_factor, last_sampling = windows[1]
        last = layers_b["HVHV"]
        if (epsg != last_epsg or transform != last_transform or first.shape != last.shape
                or factor != last_factor or sampling != last_sampling or not np.array_equal(area_mask, last_mask)):
            raise AnalysisError("The grids of the two dates do not match; they cannot be compared without resampling.")
        with np.errstate(divide="ignore", invalid="ignore"):
            last_db = 10 * np.log10(last)
        difference = last_db - first_db
        valid = np.isfinite(difference) & area_mask
        if not valid.any():
            raise AnalysisError("There are no valid pixels in common inside the selected area.")
        values = difference[valid]
        gap_days = (date.fromisoformat(result["dates"][1]) - date.fromisoformat(result["dates"][0])).days
        result.update({
            "valid_percent": float(valid.sum() / area_mask.sum() * 100),
            "median_change_db": float(np.median(values)),
            "decrease_percent": float(np.mean(values < -3) * 100),
            "increase_percent": float(np.mean(values > 3) * 100),
            "gap_days": gap_days,
            "images": {
                "earlier": image_data(first_db, -30, -10, "gray"),
                "later": image_data(last_db, -30, -10, "gray"),
                "change": image_data(difference, -5, 5, "RdBu"),
            },
        })

    # --- categorías: agua y bosque (necesitan HH además de HV)
    has_hh = "HHHH" in layers_a and (layers_b is None or "HHHH" in layers_b)
    if not has_hh:
        result["categories"] = None
        result["categories_note"] = "This file does not include the HHHH layer; the categories cannot be computed."
        return result
    inputs = {"hh": layers_a["HHHH"], "hv": layers_a["HVHV"], "looks": layers_a.get("numberOfLooks")}
    if layers_b is not None:
        inputs.update({"hh2": layers_b["HHHH"], "hv2": layers_b["HVHV"], "looks2": layers_b.get("numberOfLooks")})
    built = parameters.build_categories(inputs, mode, gap_days, pixel_area_km2)
    result["categories"] = {name: category_payload(data) for name, data in built.items()}
    result["pixel_area_km2"] = pixel_area_km2
    result["has_looks"] = inputs["looks"] is not None
    result["thresholds"] = {
        "water_hv_db": parameters.WATER_HV_DB, "water_hh_db": parameters.WATER_HH_DB,
        "dense_hv_db": parameters.DENSE_HV_DB, "loss_db": parameters.LOSS_DB,
        "disturbance_db": parameters.DISTURB_DB,
    }
    return result
