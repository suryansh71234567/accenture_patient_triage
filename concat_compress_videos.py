#!/usr/bin/env python3
"""
Video Concatenator and Target-Size Compressor
=============================================
Concatenates video files with automatic normalization of resolutions, aspect ratios,
and audio streams, and optionally compresses the resulting video to fit a specified
target file size in Megabytes (MB) using accurate 2-pass x264 encoding.

Usage:
  python concat_compress_videos.py video1.mp4 video2.mp4 -o output.mp4 --max-size-mb 25
  python concat_compress_videos.py --interactive
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path

# Ensure UTF-8 output encoding if supported by terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def check_ffmpeg_installed():
    """Verify that ffmpeg and ffprobe are available in the system PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        print("[ERROR] ffmpeg or ffprobe was not found in your PATH.")
        print("Please ensure FFmpeg is installed and added to your system environment variables.")
        sys.exit(1)
    return ffmpeg_path, ffprobe_path


def get_video_metadata(video_path):
    """
    Extract video metadata (duration, width, height, fps, has_audio, size)
    using ffprobe JSON output.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    
    # Extract duration
    duration = float(info.get("format", {}).get("duration", 0.0))
    file_size_bytes = int(info.get("format", {}).get("size", os.path.getsize(video_path)))
    
    # Analyze streams
    has_video = False
    has_audio = False
    width = 1920
    height = 1080
    fps = 30.0

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and not has_video:
            has_video = True
            width = int(stream.get("width", 1920))
            height = int(stream.get("height", 1080))
            # Calculate FPS
            r_fps = stream.get("r_frame_rate", "30/1")
            if "/" in r_fps:
                num, den = r_fps.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(r_fps)
        elif stream.get("codec_type") == "audio":
            has_audio = True

    return {
        "path": video_path,
        "duration": duration,
        "size_bytes": file_size_bytes,
        "size_mb": file_size_bytes / (1024 * 1024),
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": has_audio,
        "has_video": has_video
    }


def clean_path_input(raw_input: str) -> str:
    """Clean user input paths (strip quotes, whitespace, escape characters)."""
    p = raw_input.strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    return p.strip()


def build_filter_graph(videos_meta, target_width, target_height, target_fps=30, include_audio=True):
    """
    Builds a robust FFmpeg filter graph that:
    1. Scales and letterboxes/pillarboxes all videos into target_width x target_height.
    2. Sets uniform framerate and SAR (sample aspect ratio).
    3. Synthesizes silent audio if any input video lacks an audio track (when include_audio=True).
    4. Concat-muxes all synchronized streams.
    """
    filter_parts = []
    concat_inputs = []

    for i, meta in enumerate(videos_meta):
        # Video standardization filter
        v_label = f"v{i}"
        v_filter = (
            f"[{i}:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={target_fps}[{v_label}]"
        )
        filter_parts.append(v_filter)
        
        if include_audio:
            a_label = f"a{i}"
            if meta["has_audio"]:
                # Resample audio to standard 48kHz stereo
                a_filter = f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[{a_label}]"
                filter_parts.append(a_filter)
            else:
                # Generate silent audio stream matching the video duration
                dur = max(meta["duration"], 0.1)
                a_filter = f"anullsrc=channel_layout=stereo:sample_rate=48000:d={dur}[{a_label}]"
                filter_parts.append(a_filter)

            concat_inputs.append(f"[{v_label}][{a_label}]")
        else:
            concat_inputs.append(f"[{v_label}]")

    n = len(videos_meta)
    if include_audio:
        concat_filter = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
    else:
        concat_filter = "".join(concat_inputs) + f"concat=n={n}:v=1:a=0[outv]"
        
    filter_parts.append(concat_filter)
    return ";".join(filter_parts)


def concat_and_compress(video_paths, output_path, max_size_mb=None, resolution=None, preset="medium", crf=23):
    """
    Main function to concatenate and optionally compress videos to fit max_size_mb.
    """
    check_ffmpeg_installed()

    if len(video_paths) < 2:
        print("[ERROR] At least 2 video files are required to concatenate.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(">> VIDEO CONCATENATION & COMPRESSION")
    print("=" * 60)

    # 1. Analyze inputs
    print("\n[1/4] Analyzing input videos...")
    videos_meta = []
    total_duration = 0.0
    total_input_size_mb = 0.0
    max_w, max_h = 0, 0

    for i, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            print(f"[ERROR] File not found: {vp}")
            sys.exit(1)
        meta = get_video_metadata(vp)
        videos_meta.append(meta)
        total_duration += meta["duration"]
        total_input_size_mb += meta["size_mb"]
        max_w = max(max_w, meta["width"])
        max_h = max(max_h, meta["height"])

        print(f"  * Video {i+1}: {os.path.basename(vp)}")
        print(f"    - Duration  : {meta['duration']:.2f}s")
        print(f"    - Size      : {meta['size_mb']:.2f} MB")
        print(f"    - Resolution: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps")
        print(f"    - Audio     : {'Yes' if meta['has_audio'] else 'No (Silent track will be synthesized)'}")

    print(f"\n  Total Duration : {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"  Combined Size  : {total_input_size_mb:.2f} MB")

    # 2. Determine target resolution
    if resolution:
        try:
            target_w, target_h = map(int, resolution.lower().split("x"))
        except Exception:
            print(f"[WARNING] Invalid resolution '{resolution}'. Defaulting to {max_w}x{max_h}.")
            target_w, target_h = max_w, max_h
    else:
        # Match maximum dimension of inputs (ensure even dimensions)
        target_w = max_w if max_w % 2 == 0 else max_w + 1
        target_h = max_h if max_h % 2 == 0 else max_h + 1

    # 3. Target Bitrate / Size Planning
    use_2pass = False
    video_bitrate_kbps = None
    audio_bitrate_kbps = 128

    if max_size_mb is not None and max_size_mb > 0:
        use_2pass = True
        print(f"\n[2/4] Calculating bitrate for target size: {max_size_mb:.2f} MB")
        
        # 4% container / muxing overhead margin
        target_bytes = max_size_mb * 1024 * 1024
        usable_bytes = target_bytes * 0.96
        total_bitrate_bps = (usable_bytes * 8) / max(total_duration, 1.0)
        total_bitrate_kbps = total_bitrate_bps / 1000.0

        # Adjust audio bitrate based on available bandwidth
        if total_bitrate_kbps < 200:
            audio_bitrate_kbps = 48
            if target_h > 480:
                print("  [INFO] Auto-downscaling to 854x480 to preserve quality at very low target bitrate.")
                target_w, target_h = 854, 480
        elif total_bitrate_kbps < 500:
            audio_bitrate_kbps = 64
            if target_h > 720:
                print("  [INFO] Auto-downscaling to 1280x720 to preserve quality at lower target bitrate.")
                target_w, target_h = 1280, 720
        elif total_bitrate_kbps < 1000:
            audio_bitrate_kbps = 96
        else:
            audio_bitrate_kbps = 128

        video_bitrate_kbps = max(int(total_bitrate_kbps - audio_bitrate_kbps), 40)
        
        print(f"  * Target File Size  : {max_size_mb:.2f} MB")
        print(f"  * Computed Total BR : {total_bitrate_kbps:.1f} kbps")
        print(f"  * Video Bitrate     : {video_bitrate_kbps} kbps")
        print(f"  * Audio Bitrate     : {audio_bitrate_kbps} kbps")
        print(f"  * Canvas Resolution : {target_w}x{target_h}")
        print("  * Mode              : 2-Pass Target Bitrate Encoding")
    else:
        print(f"\n[2/4] Encoding Mode: Single-Pass Quality (CRF {crf}, Preset: {preset})")
        print(f"  * Canvas Resolution : {target_w}x{target_h}")

    # Base input arguments for ffmpeg
    input_args = []
    for vp in video_paths:
        input_args.extend(["-i", str(vp)])

    # Output directory creation
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 4. Execute FFmpeg
    print("\n[3/4] Processing videos...")
    
    if use_2pass:
        # Build filter complex for pass 1 (video only)
        filter_complex_pass1 = build_filter_graph(videos_meta, target_w, target_h, target_fps=30, include_audio=False)
        filter_complex_pass2 = build_filter_graph(videos_meta, target_w, target_h, target_fps=30, include_audio=True)

        # Pass 1
        pass_logfile = os.path.join(tempfile.gettempdir(), f"ffmpeg2pass_{os.getpid()}")
        pass1_cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex_pass1,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-b:v", f"{video_bitrate_kbps}k",
            "-maxrate", f"{int(video_bitrate_kbps * 1.5)}k",
            "-bufsize", f"{int(video_bitrate_kbps * 2.0)}k",
            "-preset", preset,
            "-pass", "1",
            "-passlogfile", pass_logfile,
            "-an",
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null"
        ]

        print("  [Pass 1/2] Analyzing video stream...")
        res1 = subprocess.run(pass1_cmd, capture_output=True, text=True)
        if res1.returncode != 0:
            print("[ERROR] FFmpeg Pass 1 failed!")
            print(res1.stderr)
            sys.exit(1)

        # Pass 2
        pass2_cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex_pass2,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-b:v", f"{video_bitrate_kbps}k",
            "-maxrate", f"{int(video_bitrate_kbps * 1.5)}k",
            "-bufsize", f"{int(video_bitrate_kbps * 2.0)}k",
            "-preset", preset,
            "-pass", "2",
            "-passlogfile", pass_logfile,
            "-c:a", "aac",
            "-b:a", f"{audio_bitrate_kbps}k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]

        print("  [Pass 2/2] Encoding output video and muxing audio...")
        res2 = subprocess.run(pass2_cmd, capture_output=True, text=True)
        
        # Cleanup pass log files
        for suffix in ["-0.log", "-0.log.mbtree", ".log", "-0.log.2pass.log"]:
            fpath = f"{pass_logfile}{suffix}"
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        if res2.returncode != 0:
            print("[ERROR] FFmpeg Pass 2 failed!")
            print(res2.stderr)
            sys.exit(1)

    else:
        filter_complex_single = build_filter_graph(videos_meta, target_w, target_h, target_fps=30, include_audio=True)
        single_cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex_single,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", "aac",
            "-b:a", f"{audio_bitrate_kbps}k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]

        print("  [Encoding] Processing single-pass high-quality video...")
        res = subprocess.run(single_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("[ERROR] FFmpeg processing failed!")
            print(res.stderr)
            sys.exit(1)

    # 5. Summary & Verification
    print("\n[4/4] Verification & Results")
    if os.path.exists(output_path):
        out_size_bytes = os.path.getsize(output_path)
        out_size_mb = out_size_bytes / (1024 * 1024)
        print(f"  [OK] Output successfully created at: {os.path.abspath(output_path)}")
        print(f"  [OK] Final File Size : {out_size_mb:.2f} MB")
        if max_size_mb:
            diff = max_size_mb - out_size_mb
            pct = (out_size_mb / max_size_mb) * 100
            print(f"  [OK] Target Size     : {max_size_mb:.2f} MB ({pct:.1f}% of limit, margin: {diff:.2f} MB)")
    else:
        print("[ERROR] Output file was not generated.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SUCCESS: Video concatenation and compression complete!")
    print("=" * 60 + "\n")


def interactive_mode():
    """Interactive CLI prompting user for files and compression options."""
    print("\n" + "=" * 60)
    print("INTERACTIVE VIDEO CONCATENATOR & COMPRESSOR")
    print("=" * 60)

    # Prompt video 1
    while True:
        v1_raw = input("\n[1] Enter path to First Video (or drag & drop here): ")
        v1 = clean_path_input(v1_raw)
        if v1 and os.path.isfile(v1):
            break
        print("  [X] File not found. Please enter a valid file path.")

    # Prompt video 2
    while True:
        v2_raw = input("\n[2] Enter path to Second Video (or drag & drop here): ")
        v2 = clean_path_input(v2_raw)
        if v2 and os.path.isfile(v2):
            break
        print("  [X] File not found. Please enter a valid file path.")

    # Optional extra videos
    extra_videos = []
    while True:
        add_more = input("\n[+] Add another video? (y/N): ").strip().lower()
        if add_more == "y":
            ve_raw = input("    Enter path to additional video: ")
            ve = clean_path_input(ve_raw)
            if ve and os.path.isfile(ve):
                extra_videos.append(ve)
            else:
                print("    [X] File not found. Skipped.")
        else:
            break

    all_videos = [v1, v2] + extra_videos

    # Prompt target size
    max_size_mb = None
    size_input = input("\n[3] Enter Target File Size in MB (e.g., 25, 50, 100) or press [ENTER] to skip compression: ").strip()
    if size_input:
        try:
            max_size_mb = float(size_input)
        except ValueError:
            print("  [!] Invalid number. Proceeding without target file size limit.")

    # Prompt output file
    default_out = "merged_output.mp4"
    out_input = input(f"\n[4] Enter Output File Path [Default: {default_out}]: ").strip()
    output_path = clean_path_input(out_input) if out_input else default_out
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    concat_and_compress(
        video_paths=all_videos,
        output_path=output_path,
        max_size_mb=max_size_mb
    )


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate two or more video files and optionally compress to a given target size in MB."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help="Paths to input video files (e.g. video1.mp4 video2.mp4)"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged_output.mp4",
        help="Path for output video (default: merged_output.mp4)"
    )
    parser.add_argument(
        "-s", "--max-size-mb",
        type=float,
        default=None,
        help="Target maximum file size in Megabytes (MB) (e.g. 25 for Discord/Slack limits)"
    )
    parser.add_argument(
        "-r", "--resolution",
        default=None,
        help="Canvas resolution (e.g., 1920x1080, 1280x720). Defaults to largest input resolution."
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="x264 encoding preset (default: medium)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Constant Rate Factor for single-pass mode (0-51, default: 23, lower=higher quality)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode with user prompts"
    )

    args = parser.parse_args()

    # If no video arguments are passed or interactive flag is set, run interactive mode
    if args.interactive or len(args.videos) < 2:
        if not args.interactive and len(args.videos) == 0:
            interactive_mode()
        elif not args.interactive and len(args.videos) == 1:
            print("[ERROR] Please provide at least 2 videos to concatenate, or run with --interactive.")
            sys.exit(1)
        else:
            interactive_mode()
    else:
        concat_and_compress(
            video_paths=args.videos,
            output_path=args.output,
            max_size_mb=args.max_size_mb,
            resolution=args.resolution,
            preset=args.preset,
            crf=args.crf
        )


if __name__ == "__main__":
    main()
