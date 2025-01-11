#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from os import path

###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        prog='webrtc-vmaf-variant',
        description="Encode input video using time-varying bitrates from CSV, then compute VMAF."
    )
    parser.add_argument('input', help="Path to the input reference video file")
    parser.add_argument('--codec', default='h264', help="Codec to use (h264, vp8, vp9, av1, etc.)")
    parser.add_argument('--framerate', default=30, type=int, help="Frame rate for encoding")
    parser.add_argument('--width', type=int, help="Width to encode (if omitted, use source)")
    parser.add_argument('--height', type=int, help="Height to encode (if omitted, use source)")
    parser.add_argument('--bitrate-trace', required=True, help="CSV with one bitrate per row")
    parser.add_argument('--segment-duration', type=float, default=1.0,
                        help="Segment duration in seconds (defaults to 1s per CSV row)")
    parser.add_argument('--output', default="final_variant_output.mkv",
                        help="Name of the final stitched output file")

    args = parser.parse_args()

    # Create parent directory if it doesn't exist
    parent_dir = 'tmp_vmaf_variant'
    os.makedirs(parent_dir, exist_ok=True)

    # Create timestamp-based subdirectory
    timestamp = int(time.time())
    temp_dir = os.path.join(parent_dir, str(timestamp))
    os.makedirs(temp_dir, exist_ok=True)
    print(f"[INFO] Created temporary directory: {temp_dir}")

    # Read the per-second bitrates from CSV
    bitrates = read_bitrate_trace(args.bitrate_trace)
    if not bitrates:
        print(f"[ERROR] No valid bitrates found in {args.bitrate_trace}")
        sys.exit(1)

    # Gather info about the reference video
    ref_w, ref_h, ref_duration, _ = get_video_info(args.input, None, None)
    print(f"[INFO] Input video: {args.input}")
    print(f"       Resolution: {ref_w}x{ref_h}, Duration: {ref_duration:.2f}s")

    # Use input resolution if provided, else match source
    encode_w = args.width if args.width else ref_w
    encode_h = args.height if args.height else ref_h

    # Create a CSV to store all VMAF data (frame-level).
    # We'll append rows for each segment.
    vmaf_log_file = os.path.join(temp_dir, "vmaf_segment_scores.csv")
    with open(vmaf_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_id', 'bitrate', 'encoded_res', 'frame_number', 'vmaf_score'])

    print(f"[INFO] Splitting '{args.input}' into ~{args.segment_duration:.1f}s segments.")
    total_segments = int(math.ceil(ref_duration / args.segment_duration))

    segment_paths = []
    all_frames_vmaf = []  # We'll store ALL segments' frame scores here
    seg_idx = 0

    # Process each segment until we exhaust the bitrate trace
    while seg_idx < len(bitrates):
        # Calculate segment start time (or simply seg_idx * duration, etc.)
        seg_start = (seg_idx * args.segment_duration) % ref_duration
        bitrate = bitrates[seg_idx]

        # Calculate appropriate resolution for this bitrate
        encode_w, encode_h = calculate_resolution(ref_w, ref_h, bitrate, args.framerate)

        print(f"\n[INFO] Segment {seg_idx}:")
        print(f"  Source: {ref_w}x{ref_h}")
        print(f"  Target: {encode_w}x{encode_h}")
        print(f"  Bitrate: {bitrate}Kbps")
        print(f"  Time: {seg_start:.2f}s -> {seg_start + args.segment_duration:.2f}s")

        try:
            # Extract segment from reference
            seg_in = os.path.join(temp_dir, f"seg_in_{seg_idx}.mkv")
            extract_segment(args.input, seg_in, seg_start, args.segment_duration)

            # Encode segment
            seg_out = os.path.join(
                temp_dir,
                f"seg_encoded_{args.codec}_{encode_w}x{encode_h}_{bitrate}_{seg_idx}.mkv"
            )
            encode_file(
                seg_in,
                seg_out,
                codec=args.codec,
                width=encode_w,
                height=encode_h,
                bitrate=bitrate,
                framerate=args.framerate,
            )
            
            # Create upscaled version for VMAF comparison
            seg_upscaled = os.path.join(temp_dir, f"seg_upscaled_{seg_idx}.mkv")
            upscale_segment(seg_out, seg_upscaled, ref_w, ref_h)

            segment_paths.append(seg_upscaled)

            # -----------------------------------------------------------
            # Compute frame-by-frame VMAF for this segment
            # -----------------------------------------------------------
            vmaf_scores = compute_vmaf_segment(
                ref_file=seg_in,          # The *reference* is the cut piece (seg_in)
                dist_file=seg_upscaled,   # The *distorted* is the upscaled version
                ref_width=ref_w,
                ref_height=ref_h,
                dist_width=ref_w,
                dist_height=ref_h,
                framerate=args.framerate,
                temp_dir=temp_dir
            )

            # Append these frame scores to our global list
            all_frames_vmaf.extend(vmaf_scores)

            # Write each frame's VMAF to CSV
            encoded_res_str = f"{encode_w}x{encode_h}"
            with open(vmaf_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for frame_num, score in enumerate(vmaf_scores):
                    writer.writerow([seg_idx, bitrate, encoded_res_str, frame_num, score])

            # Print stats for this segment
            if len(vmaf_scores) > 0:
                seg_min = min(vmaf_scores)
                seg_max = max(vmaf_scores)
                seg_mean = sum(vmaf_scores) / len(vmaf_scores)
                seg_std = math.sqrt(sum((s - seg_mean) ** 2 for s in vmaf_scores) / len(vmaf_scores))
                print(f"  VMAF Stats for segment {seg_idx}:")
                print(f"    Min VMAF:  {seg_min:.2f}")
                print(f"    Max VMAF:  {seg_max:.2f}")
                print(f"    Mean VMAF: {seg_mean:.2f}")
                print(f"    Std Dev:   {seg_std:.2f}")
            else:
                print(f"[WARNING] No VMAF frames computed for segment {seg_idx}?!")

        except Exception as e:
            print(f"[ERROR] Failed processing segment {seg_idx}: {str(e)}")
            # Optionally break or continue based on your needs
            raise

        seg_idx += 1

    # -----------------------------------------------------------
    # Concatenate all segments into one final output
    # -----------------------------------------------------------
    print(f"[INFO] Concatenating {len(segment_paths)} segments into '{args.output}'")
    concat_segments(segment_paths, args.output)

    # -----------------------------------------------------------
    # (Optional) Compute overall stats across **all frames** 
    # from **all segments** that we processed
    # -----------------------------------------------------------
    if all_frames_vmaf:
        overall_min = min(all_frames_vmaf)
        overall_max = max(all_frames_vmaf)
        overall_mean = sum(all_frames_vmaf) / len(all_frames_vmaf)
        overall_std = math.sqrt(
            sum((s - overall_mean) ** 2 for s in all_frames_vmaf) / len(all_frames_vmaf)
        )

        print("\n[INFO] Overall VMAF Stats (across all segments, all frames):")
        print(f"    Min VMAF:  {overall_min:.2f}")
        print(f"    Max VMAF:  {overall_max:.2f}")
        print(f"    Mean VMAF: {overall_mean:.2f}")
        print(f"    Std Dev:   {overall_std:.2f}")
    else:
        print("\n[WARNING] No overall VMAF frames computed?!")

    # Print final output locations
    print("\n[INFO] Output Locations:")
    print(f"    Working Directory: {temp_dir}")
    print(f"    VMAF Log File:     {vmaf_log_file}")
    print(f"    Final Output:      {args.output}")


###############################################################################
# HELPERS
###############################################################################

def compute_vmaf_segment(ref_file, dist_file,
                         ref_width, ref_height,
                         dist_width, dist_height,
                         framerate=30,
                         temp_dir=None):
    """
    Compute VMAF scores for a segment and return frame-by-frame scores.
    Returns a list of VMAF scores for each frame.
    """
    # Create a temp JSON file for VMAF scores
    if temp_dir:
        temp_json = os.path.join(temp_dir, f"vmaf_scores_{int(time.time()*1000)}.json")
    else:
        temp_json = "vmaf_scores.json"
    temp_json = os.path.abspath(temp_json)

    filter_str = (
        f'[0:v]fps={framerate},settb=AVTB,setpts=PTS-STARTPTS[ref];'
        f'[1:v]fps={framerate},scale={dist_width}:{dist_height}:flags=bicubic,'
        'settb=AVTB,setpts=PTS-STARTPTS[dist];'
        f'[dist][ref]libvmaf=n_threads=8:log_path={temp_json}:log_fmt=json'
    )

    command = [
        'ffmpeg',
        '-i', ref_file,
        '-i', dist_file,
        '-filter_complex', filter_str,
        '-f', 'null',
        '-'
    ]

    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"[ERROR] FFmpeg for VMAF failed: {process.stderr}")
        return []

    # Read and parse the JSON file
    frame_scores = []
    try:
        with open(temp_json, 'r') as f:
            vmaf_data = json.load(f)
            frame_scores = [frame["metrics"]["vmaf"] for frame in vmaf_data["frames"]]
    except Exception as e:
        print(f"[ERROR] Failed to read VMAF scores from {temp_json}: {e}")

    return frame_scores

def calculate_resolution(original_width, original_height, bitrate, framerate):
    """
    Select an appropriate resolution from standard options based on the bitrate,
    but do not exceed the original video's resolution. 
    Available standard resolutions: 640x360 (360p), 1280x720 (720p), 
    1920x1080 (1080p), 3840x2160 (4K).
    
    Returns (width, height) pair.
    """
    # Standard resolutions
    RESOLUTIONS = [
        (640, 360),     # 360p
        (1280, 720),    # 720p
        (1920, 1080),   # 1080p
        (3840, 2160)    # 4K
    ]
    
    # Rough bitrate thresholds in Kbps for each resolution
    # Using ~0.05 bits per pixel at given framerate as an example
    bitrate_thresholds = [
        640 * 360   * framerate * 0.05 / 1000,   # For 360p
        1280 * 720  * framerate * 0.05 / 1000,   # For 720p
        1920 * 1080 * framerate * 0.05 / 1000,   # For 1080p
        3840 * 2160 * framerate * 0.05 / 1000    # For 4K
    ]
    
    # Filter out any resolution that exceeds the original
    valid_pairs = []
    for (res_w, res_h), threshold in zip(RESOLUTIONS, bitrate_thresholds):
        if res_w <= original_width and res_h <= original_height:
            valid_pairs.append(((res_w, res_h), threshold))
    
    # If no valid resolution is smaller than or equal to original,
    # default to the smallest standard resolution
    if not valid_pairs:
        valid_pairs = [((640, 360), bitrate_thresholds[0])]
    
    # Default to the smallest valid resolution
    selected_width, selected_height = valid_pairs[0][0]
    
    # Choose the highest resolution (among valid ones) that meets the bitrate threshold
    for (res_w, res_h), threshold in valid_pairs:
        if bitrate >= threshold:
            selected_width, selected_height = res_w, res_h
    
    print(f"[DEBUG] Resolution selection:")
    print(f"  Original: {original_width}x{original_height}")
    print(f"  Input: {bitrate} Kbps at {framerate} fps")
    print("  Valid resolutions under original:", 
          ", ".join(f"{w}x{h}" for (w, h), _ in valid_pairs))
    print("  Thresholds (Kbps):", 
          ", ".join(f"{w}x{h}:{int(th)}" for ((w, h), th) in valid_pairs))
    print(f"  Selected: {selected_width}x{selected_height}")
    actual_bpp = (bitrate * 1000)/(selected_width * selected_height * framerate)
    print(f"  Actual bits/pixel: {actual_bpp:.3f}")
    
    return selected_width, selected_height

def upscale_segment(input_file, output_file, target_width, target_height):
    """
    Upscale a video segment to target resolution for fair VMAF comparison
    """
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'scale={target_width}x{target_height}:flags=bicubic',
        '-c:v', 'libx264',  # Use H.264 for the upscaled version
        '-crf', '0',        # Lossless to avoid additional quality loss
        '-preset', 'fast',
        '-y',
        output_file
    ]
    run_ffmpeg(command, desc="upscale_segment")

def read_bitrate_trace(csv_file):
    """
    Reads bitrates from a CSV. 
    Expects a header row with a column named 'bitrates'.
    Returns a list of integers (bitrate in Kbps).
    """
    bitrates = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # Find bitrates column index
        try:
            bitrate_col = headers.index('bitrates')
            print("[INFO] Found 'bitrates' column in CSV.")
        except ValueError:
            print("[ERROR] No 'bitrates' column found in CSV header.")
            return []
            
        for row in reader:
            if not row:
                continue
            try:
                # read from correct column and convert to int Kbps
                br = float(row[bitrate_col]) / 1000  # CSV might have bits or bps, adjust as needed
                if br > 0:
                    bitrates.append(int(br))
            except (ValueError, IndexError):
                continue
                
    if len(bitrates) > 0:
        print(f"[INFO] Read {len(bitrates)} valid bitrate values")
        print(f"[INFO] Bitrate range: {min(bitrates):.0f} - {max(bitrates):.0f} Kbps")
    else:
        print(f"[WARNING] No valid bitrates in {csv_file}")

    return bitrates

def extract_segment(input_file, output_file, start, duration):
    """
    Extract a segment [start, start+duration] from input_file -> output_file.
    For Y4M input, re-encode to a lossless intermediate; for other formats, we try -c copy.
    """
    # Check if input is Y4M
    is_y4m = input_file.lower().endswith('.y4m')

    if is_y4m:
        # Must re-encode raw Y4M to a container format
        command = [
            'ffmpeg',
            '-ss', f'{start:.3f}',
            '-i', input_file,
            '-t', f'{duration:.3f}',
            '-y',
            '-loglevel', 'error',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-qp', '0',
            '-pix_fmt', 'yuv420p',
            output_file
        ]
    else:
        # For typical container files, we can try to copy the segment
        command = [
            'ffmpeg',
            '-ss', f'{start:.3f}',
            '-i', input_file,
            '-t', f'{duration:.3f}',
            '-c', 'copy',
            '-y',
            '-loglevel', 'error',
            output_file
        ]

    run_ffmpeg(command, desc="extract_segment")

def concat_segments(segment_paths, output_file):
    """
    Concatenates all segments in `segment_paths` into `output_file` using FFmpeg.
    """
    if not segment_paths:
        print("[WARNING] No segments to concatenate!")
        return

    list_file = "tmp_vmaf_variant/concat_list.txt"
    with open(list_file, 'w') as f:
        for seg_path in segment_paths:
            f.write(f"file '{os.path.abspath(seg_path)}'\n")

    command = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        '-loglevel', 'error',
        output_file
    ]
    run_ffmpeg(command, desc="concat_segments")

def get_video_info(input_file, width, height):
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_format',
        '-show_streams',
        '-select_streams', 'v:0',
        '-of', 'json',
        input_file
    ]
    process = subprocess.run(command, capture_output=True, text=True)

    if process.stderr:
        err = process.stderr.strip()
        if ("No such file" in err) or ("error" in err.lower()):
            raise RuntimeError(f"[ffprobe error] {err}")

    output_json = json.loads(process.stdout or "{}")
    if 'streams' not in output_json or len(output_json['streams']) == 0:
        raise RuntimeError("[ffprobe error] Could not find video stream info.")

    # Use input width/height if provided, else from the file
    if not width:
        width = output_json['streams'][0]['width']
    if not height:
        height = output_json['streams'][0]['height']

    duration = 0.0
    bitrate = 0
    stream_duration = output_json['streams'][0].get('duration', None)
    format_duration = output_json['format'].get('duration', None)
    if stream_duration:
        duration = float(stream_duration)
    elif format_duration:
        duration = float(format_duration)

    if 'bit_rate' in output_json['format']:
        bitrate = int(output_json['format']['bit_rate'])

    return width, height, duration, bitrate

def encode_file(input_file, output_file, codec, width, height, bitrate, framerate):
    """
    Encodes input_file to output_file using the chosen codec, resolution, and bitrate.
    """
    bitrate_str = f'{bitrate}K'
    filters = f'fps={framerate},scale={width}x{height}:flags=bicubic,format=yuv420p'

    # Determine segment duration from ffprobe
    duration_str = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]).decode().strip()

    try:
        duration = float(duration_str)
    except ValueError:
        duration = 1.0  # fallback if we can't parse

    # Rough attempt to limit threads based on resolution
    threads = max(1, int((width * height) / (640 * 480)))  # scale as you wish

    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter:v', filters,
        '-threads', f'{threads}',
        '-an',   # discard audio
        '-y',
        '-loglevel', 'error',
        '-c:v'
    ]

    if codec == 'h264':
        command.extend([
            'libx264',
            '-preset', 'veryfast',
            '-rc-lookahead', '0',
            '-profile:v', 'baseline',
            '-maxrate', bitrate_str,
            '-minrate', bitrate_str,
            '-bufsize', bitrate_str,
        ])
    elif codec == 'h264_zerolatency':
        command.extend([
            'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-maxrate', bitrate_str,
            '-minrate', bitrate_str,
            '-bufsize', bitrate_str,
        ])
    elif codec == 'vp8':
        command.extend([
            'libvpx',
            '-b:v', bitrate_str,
            '-maxrate', bitrate_str,
            '-minrate', bitrate_str,
            '-bufsize', bitrate_str,
            '-crf', '10',
            '-quality', 'good',
            '-cpu-used', '5',
            '-rc_lookahead', '0',
            '-lag-in-frames', '0',
            '-g', '120',
            '-deadline', 'good',
        ])
    elif codec == 'vp9':
        command.extend([
            'libvpx-vp9',
            '-b:v', bitrate_str,
            '-minrate', bitrate_str,
            '-maxrate', bitrate_str,
            '-bufsize', bitrate_str,
            '-crf', '10',
            '-quality', 'good',
            '-cpu-used', '3',
            '-rc_lookahead', '0',
            '-lag-in-frames', '0',
            '-g', '120',
            '-deadline', 'good',
            '-row-mt', '1',
            '-tile-columns', '2',
            '-tile-rows', '1',
        ])
    elif codec == 'av1':
        command.extend([
            'libaom-av1',
            '-b:v', bitrate_str,
            '-usage', 'realtime',
            '-minrate', bitrate_str,
            '-maxrate', bitrate_str,
            '-bufsize', bitrate_str,
            '-cpu-used', '8',
            '-row-mt', '1',
            '-qmax', '52',
            '-qmin', '10',
            '-aq-mode', '3',
            '-enable-global-motion', '0',
            '-enable-intrabc', '0',
            '-enable-restoration', '0',
            '-enable-interintra-comp', '0',
            '-enable-interintra-wedge', '0',
            '-refs', '3',
        ])
        if height >= 360:
            command.extend(['-tile-columns', '3'])
    else:
        raise ValueError(f"Unsupported codec '{codec}'")

    command.append(output_file)
    run_ffmpeg(command, desc=f"encode_file at {bitrate}K")

    # Optional: compute how close we got to target bitrate
    output_size = os.path.getsize(output_file)
    actual_bitrate = (output_size * 8 / 1000) / max(0.0001, duration)
    print(f"[DEBUG] Target bitrate:  {bitrate} Kbps")
    print(f"[DEBUG] Actual bitrate: {actual_bitrate:.2f} Kbps")

def run_ffmpeg(command, desc=""):
    """
    Helper that runs ffmpeg and raises an exception on error.
    """
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        msg = f"[ERROR] FFmpeg failed ({desc}): {process.stderr}"
        raise RuntimeError(msg)


if __name__ == '__main__':
    main()
