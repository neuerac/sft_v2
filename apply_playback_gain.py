# coding=utf-8
"""Apply post-generation playback gain without changing any SFT control text.

The tool operates on a rendered audio file, never on model inputs or training
data.  It supports either a fixed gain or a target integrated loudness, then
uses a conservative global attenuation pass to keep the rendered file below a
configurable true-peak estimate.

``pyloudnorm`` is required for ``--target-lufs``.  It is optional for a fixed
``--gain-db`` request: the report will still contain RMS, peak, and clipping
metrics, with ``lufs_i`` set to ``null`` when the dependency is unavailable.
``scipy`` is optional; when installed, ``resample_poly`` is used for an
oversampled true-peak estimate.  Otherwise the script falls back to sample
peak and records that limitation in the metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_TRUE_PEAK_DBFS = -1.0


class DependencyError(RuntimeError):
    """Raised when a requested audio operation needs an unavailable package."""


def require_numpy():
    """Import NumPy only when audio processing actually starts."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise DependencyError("missing dependency 'numpy'; install it with: pip install numpy") from exc
    return np


def require_soundfile():
    """Import SoundFile only when an input or output file is accessed."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise DependencyError(
            "missing dependency 'soundfile'; install it with: pip install soundfile"
        ) from exc
    return sf


def optional_pyloudnorm():
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None
    return pyln


def require_pyloudnorm():
    pyln = optional_pyloudnorm()
    if pyln is None:
        raise DependencyError(
            "--target-lufs requires 'pyloudnorm'; install it with: pip install pyloudnorm"
        )
    return pyln


def optional_scipy_signal():
    try:
        from scipy import signal
    except ImportError:
        return None
    return signal


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def db_to_linear(gain_db: float) -> float:
    """Return a linear amplitude scale for a finite dB value."""
    gain_db = finite_float(gain_db, "gain_db")
    return 10.0 ** (gain_db / 20.0)


def linear_to_db(value: float) -> float | None:
    if not math.isfinite(value) or value <= 0.0:
        return None
    return 20.0 * math.log10(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a post-generation playback gain or target LUFS, then keep the "
            "result below a true-peak ceiling. This never adds or reads SFT tags."
        )
    )
    parser.add_argument(
        "--input",
        "--input-audio",
        dest="input_audio",
        required=True,
        help="Input audio readable by soundfile (for example WAV or FLAC).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output audio path. It must differ from --input.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--gain-db",
        type=float,
        help="Requested post-generation playback gain in dB.",
    )
    mode.add_argument(
        "--target-lufs",
        type=float,
        help="Target integrated LUFS. Requires pyloudnorm.",
    )
    parser.add_argument(
        "--true-peak-dbfs",
        type=float,
        default=DEFAULT_TRUE_PEAK_DBFS,
        help=f"Maximum estimated true peak in dBFS (default: {DEFAULT_TRUE_PEAK_DBFS}).",
    )
    parser.add_argument(
        "--true-peak-oversample",
        type=int,
        choices=(1, 2, 4, 8),
        default=4,
        help=(
            "Oversampling factor for true-peak estimation (default: 4). Values above "
            "1 use scipy when available, otherwise fall back to sample peak."
        ),
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.999,
        help="Absolute sample level counted as near-clipping in reported metrics.",
    )
    parser.add_argument(
        "--max-ceiling-passes",
        type=int,
        default=3,
        help="Maximum write/measure correction passes for the peak ceiling (default: 3).",
    )
    parser.add_argument(
        "--output-subtype",
        default=None,
        help=(
            "Optional soundfile subtype for the output, such as PCM_16 or FLOAT. "
            "By default soundfile chooses the format's default subtype."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        "--report-json",
        dest="metrics_json",
        default=None,
        help="Metrics JSON path. Defaults to <output-suffix>.metrics.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output audio file and metrics JSON.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_numeric_args(args: argparse.Namespace) -> None:
    if args.gain_db is not None:
        args.gain_db = finite_float(args.gain_db, "--gain-db")
        if not -120.0 <= args.gain_db <= 120.0:
            raise ValueError("--gain-db must be within [-120, 120] dB")
    if args.target_lufs is not None:
        args.target_lufs = finite_float(args.target_lufs, "--target-lufs")
        if not -70.0 <= args.target_lufs < 0.0:
            raise ValueError("--target-lufs must be in [-70, 0) LUFS")

    args.true_peak_dbfs = finite_float(args.true_peak_dbfs, "--true-peak-dbfs")
    if not -60.0 <= args.true_peak_dbfs < 0.0:
        raise ValueError("--true-peak-dbfs must be in [-60, 0) dBFS")

    args.clip_threshold = finite_float(args.clip_threshold, "--clip-threshold")
    if not 0.0 < args.clip_threshold <= 1.0:
        raise ValueError("--clip-threshold must be in (0, 1]")
    if args.max_ceiling_passes < 1:
        raise ValueError("--max-ceiling-passes must be >= 1")


def default_metrics_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metrics.json")


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source = Path(args.input_audio).expanduser()
    output = Path(args.output).expanduser()
    metrics = Path(args.metrics_json).expanduser() if args.metrics_json else default_metrics_path(output)

    if not source.is_file():
        raise FileNotFoundError(f"input audio does not exist or is not a file: {source}")
    if not output.suffix:
        raise ValueError("--output must include an audio filename extension, for example .wav")
    if not metrics.suffix:
        raise ValueError("--metrics-json must include a filename extension, for example .json")

    source_resolved = source.resolve()
    output_resolved = output.resolve()
    metrics_resolved = metrics.resolve()
    if output_resolved == source_resolved:
        raise ValueError("--output must differ from --input; in-place processing is not supported")
    if metrics_resolved in (source_resolved, output_resolved):
        raise ValueError("--metrics-json must differ from both --input and --output")
    if not args.overwrite:
        existing = [str(path) for path in (output, metrics) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing file(s); pass --overwrite: " + ", ".join(existing)
            )
    return source, output, metrics


def true_peak_linear(audio, oversample: int, np, scipy_signal) -> tuple[float, str]:
    """Measure an estimated inter-sample peak, with an explicit fallback method."""
    if audio.size == 0:
        return 0.0, "sample_peak_empty"
    sample_peak = float(np.max(np.abs(audio)))
    if oversample == 1:
        return sample_peak, "sample_peak"
    if scipy_signal is None:
        return sample_peak, "sample_peak_fallback_no_scipy"
    try:
        oversampled = scipy_signal.resample_poly(
            audio,
            up=oversample,
            down=1,
            axis=0,
            padtype="line",
        )
        return float(np.max(np.abs(oversampled))), f"scipy_resample_poly_{oversample}x"
    except (TypeError, ValueError, MemoryError):
        # The audio remains safe at sample level, and the report exposes that an
        # inter-sample estimate could not be obtained for this particular file.
        return sample_peak, "sample_peak_fallback_resample_failed"


def integrated_lufs(audio, sample_rate: int, pyln) -> tuple[float | None, str | None]:
    """Return integrated LUFS or an explanatory reason when it is undefined."""
    if pyln is None:
        return None, "pyloudnorm is not installed"
    if not len(audio):
        return None, "audio is empty"
    try:
        value = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        return None, f"pyloudnorm failed: {type(exc).__name__}: {exc}"
    if not math.isfinite(value):
        return None, "integrated loudness is not finite (for example, silence)"
    return value, None


def audio_metrics(
    audio,
    sample_rate: int,
    *,
    clip_threshold: float,
    true_peak_oversample: int,
    np,
    scipy_signal,
    pyln,
) -> tuple[dict[str, Any], str | None]:
    """Return JSON-safe waveform, peak, clipping, and optional LUFS metrics."""
    if audio.ndim != 2:
        raise ValueError(f"expected audio with shape [frames, channels], got {audio.shape}")
    if not audio.size:
        raise ValueError("audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or infinity")

    sample_peak = float(np.max(np.abs(audio)))
    true_peak, true_peak_method = true_peak_linear(
        audio, true_peak_oversample, np, scipy_signal
    )
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    lufs, lufs_error = integrated_lufs(audio, sample_rate, pyln)
    return (
        {
            "duration_sec": float(audio.shape[0] / sample_rate),
            "sample_rate": int(sample_rate),
            "channels": int(audio.shape[1]),
            "sample_peak_dbfs": linear_to_db(sample_peak),
            "true_peak_dbfs": linear_to_db(true_peak),
            "true_peak_method": true_peak_method,
            "rms_dbfs": linear_to_db(rms),
            "clipping_ratio": float(np.mean(np.abs(audio) >= clip_threshold)),
            "hard_clip_ratio": float(np.mean(np.abs(audio) >= 1.0)),
            "lufs_i": lufs,
        },
        lufs_error,
    )


def apply_peak_ceiling(
    audio,
    *,
    ceiling_linear: float,
    true_peak_oversample: int,
    np,
    scipy_signal,
) -> tuple[Any, float, float, str]:
    """Apply global attenuation when the true-peak estimate exceeds the ceiling."""
    peak, method = true_peak_linear(audio, true_peak_oversample, np, scipy_signal)
    if peak <= ceiling_linear or peak == 0.0:
        return audio, 0.0, peak, method
    reduction_linear = ceiling_linear / peak
    reduction_db = linear_to_db(reduction_linear)
    # reduction_linear is in (0, 1), so linear_to_db cannot return None here.
    assert reduction_db is not None
    adjusted = audio * reduction_linear
    final_peak, final_method = true_peak_linear(
        adjusted, true_peak_oversample, np, scipy_signal
    )
    return adjusted, reduction_db, final_peak, final_method


def temporary_output_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.tmp-",
        suffix=output.suffix,
        dir=str(output.parent),
    )
    os.close(descriptor)
    return Path(temporary_name)


def write_audio(
    sf,
    path: Path,
    audio,
    sample_rate: int,
    output_subtype: str | None,
) -> None:
    kwargs: dict[str, Any] = {}
    if output_subtype:
        kwargs["subtype"] = output_subtype
    try:
        sf.write(str(path), audio, sample_rate, **kwargs)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot write output audio {path}; choose a soundfile-writable extension "
            f"and compatible --output-subtype: {exc}"
        ) from exc


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def render_with_ceiling(
    sf,
    requested_audio,
    sample_rate: int,
    temporary: Path,
    *,
    true_peak_ceiling_dbfs: float,
    true_peak_oversample: int,
    max_ceiling_passes: int,
    output_subtype: str | None,
    clip_threshold: float,
    np,
    scipy_signal,
    pyln,
) -> tuple[Any, dict[str, Any], str | None, float]:
    """Render to a temporary file and correct any output-format peak overshoot."""
    ceiling_linear = db_to_linear(true_peak_ceiling_dbfs)
    current = requested_audio
    total_ceiling_reduction_db = 0.0
    postwrite_metrics: dict[str, Any] | None = None
    postwrite_lufs_error: str | None = None

    for write_pass in range(1, max_ceiling_passes + 1):
        current, reduction_db, _, _ = apply_peak_ceiling(
            current,
            ceiling_linear=ceiling_linear,
            true_peak_oversample=true_peak_oversample,
            np=np,
            scipy_signal=scipy_signal,
        )
        total_ceiling_reduction_db += reduction_db
        write_audio(sf, temporary, current, sample_rate, output_subtype)
        rendered, rendered_rate = sf.read(str(temporary), dtype="float64", always_2d=True)
        if int(rendered_rate) != int(sample_rate):
            raise RuntimeError(
                f"output writer changed sample rate from {sample_rate} to {rendered_rate}: {temporary}"
            )
        postwrite_metrics, postwrite_lufs_error = audio_metrics(
            rendered,
            int(rendered_rate),
            clip_threshold=clip_threshold,
            true_peak_oversample=true_peak_oversample,
            np=np,
            scipy_signal=scipy_signal,
            pyln=pyln,
        )
        true_peak_dbfs = postwrite_metrics["true_peak_dbfs"]
        if true_peak_dbfs is None or true_peak_dbfs <= true_peak_ceiling_dbfs + 1e-6:
            return rendered, postwrite_metrics, postwrite_lufs_error, total_ceiling_reduction_db

        # Quantization or a format conversion can shift a peak slightly. Start
        # the next pass from the file that will actually be delivered.
        current = rendered

    assert postwrite_metrics is not None
    raise RuntimeError(
        "could not enforce the requested true-peak ceiling after "
        f"{max_ceiling_passes} pass(es): ceiling={true_peak_ceiling_dbfs:.3f} dBFS, "
        f"measured={postwrite_metrics['true_peak_dbfs']} dBFS"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_numeric_args(args)
    source, output, metrics_path = validate_paths(args)
    np = require_numpy()
    sf = require_soundfile()
    pyln = require_pyloudnorm() if args.target_lufs is not None else optional_pyloudnorm()
    scipy_signal = optional_scipy_signal() if args.true_peak_oversample > 1 else None

    try:
        input_info = sf.info(str(source))
        audio, sample_rate = sf.read(str(source), dtype="float64", always_2d=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"cannot read input audio {source}: {exc}") from exc
    if not int(sample_rate):
        raise ValueError(f"input audio has an invalid sample rate: {sample_rate}")

    before, before_lufs_error = audio_metrics(
        audio,
        int(sample_rate),
        clip_threshold=args.clip_threshold,
        true_peak_oversample=args.true_peak_oversample,
        np=np,
        scipy_signal=scipy_signal,
        pyln=pyln,
    )
    if args.target_lufs is not None:
        if before["lufs_i"] is None:
            raise ValueError(
                "cannot apply --target-lufs because input integrated loudness is unavailable: "
                f"{before_lufs_error}"
            )
        requested_gain_db = float(args.target_lufs - before["lufs_i"])
        mode = "target_lufs"
    else:
        requested_gain_db = float(args.gain_db)
        mode = "gain_db"

    requested_audio = audio * db_to_linear(requested_gain_db)
    temporary = temporary_output_path(output)
    try:
        _, after, after_lufs_error, ceiling_reduction_db = render_with_ceiling(
            sf,
            requested_audio,
            int(sample_rate),
            temporary,
            true_peak_ceiling_dbfs=args.true_peak_dbfs,
            true_peak_oversample=args.true_peak_oversample,
            max_ceiling_passes=args.max_ceiling_passes,
            output_subtype=args.output_subtype,
            clip_threshold=args.clip_threshold,
            np=np,
            scipy_signal=scipy_signal,
            pyln=pyln,
        )
        report = {
            "input_audio": str(source.resolve()),
            "output_audio": str(output.resolve()),
            "metrics_json": str(metrics_path.resolve()),
            "mode": mode,
            "requested_gain_db": requested_gain_db,
            "requested_target_lufs": args.target_lufs,
            "true_peak_ceiling_dbfs": args.true_peak_dbfs,
            "true_peak_oversample_requested": args.true_peak_oversample,
            "ceiling_reduction_db": ceiling_reduction_db,
            "nominal_total_gain_db": requested_gain_db + ceiling_reduction_db,
            "measured_lufs_gain_db": (
                after["lufs_i"] - before["lufs_i"]
                if after["lufs_i"] is not None and before["lufs_i"] is not None
                else None
            ),
            "source_format": input_info.format,
            "source_subtype": input_info.subtype,
            "output_subtype_requested": args.output_subtype,
            "before": before,
            "after": after,
            "warnings": [],
        }
        if before_lufs_error:
            report["warnings"].append(f"before LUFS unavailable: {before_lufs_error}")
        if after_lufs_error:
            report["warnings"].append(f"after LUFS unavailable: {after_lufs_error}")
        if "fallback" in before["true_peak_method"] or "fallback" in after["true_peak_method"]:
            report["warnings"].append(
                "true-peak oversampling was unavailable; the ceiling was enforced against sample peak only"
            )

        # Write the report before publishing the audio file so a successful
        # output path always has its matching metrics sidecar.
        write_json_atomically(metrics_path, report)
        os.replace(temporary, output)
        return report
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
