import subprocess
import re
import json
import numpy as np
import os

def encode_video(input_file, output_file="encoded.webm", bitrate="1800k", encoder="libvpx"):
    """
    Encodes the input_file at the specified bitrate using the given encoder (default = VP8).
    """
    encode_cmd = [
        "ffmpeg",
        "-y",             # Overwrite existing files
        "-i", input_file,
        "-c:v", encoder,  # "libvpx" (VP8), "libvpx-vp9", "libx264", etc.
        "-b:v", bitrate,
        "-maxrate", bitrate,  
        "-bufsize", f"{int(bitrate[:-1]) * 2}k",  # 2x the bitrate
        output_file
    ]
    print("Running encode command:", " ".join(encode_cmd))
    subprocess.run(encode_cmd, check=True)


def calculate_ssim(reference_file, distorted_file, ssim_log="ssim.log"):
    """
    Runs the FFmpeg SSIM filter comparing reference_file vs. distorted_file.
    Returns (average_ssim, std_ssim).
    """
    ssim_cmd = [
        "ffmpeg",
        "-y",
        "-i", reference_file,
        "-i", distorted_file,
        "-lavfi", f"ssim=stats_file={ssim_log}",
        "-f", "null", "-"
    ]
    print("Running SSIM command:", " ".join(ssim_cmd))
    subprocess.run(ssim_cmd, check=True)

    all_ssim_values = []
    pattern_all = re.compile(r"All:([\d\.]+)")
    
    with open(ssim_log, "r") as f:
        for line in f:
            match_all = pattern_all.search(line)
            if match_all:
                val = float(match_all.group(1))
                all_ssim_values.append(val)

    if all_ssim_values:
        avg_ssim = float(np.mean(all_ssim_values))
        std_ssim = float(np.std(all_ssim_values))
    else:
        avg_ssim = 0.0
        std_ssim = 0.0

    return avg_ssim, std_ssim


def calculate_vmaf(
    reference_file, 
    distorted_file, 
    model_path="/mnt/mydata/home/goodsol/vmaf/model/vmaf_v0.6.1.json", 
    vmaf_log="vmaf.json"
):
    """
    Runs the FFmpeg libvmaf filter to compare reference_file vs. distorted_file.
    Returns (average_vmaf, std_vmaf).

    Since your FFmpeg build doesn't support 'model_path=...', we use the older syntax:
      libvmaf='model=path=/path/to/model.json:log_fmt=json:log_path=...'
    """
    # Use the older, single-string syntax for libvmaf
    vmaf_cmd = [
        "ffmpeg",
        "-y",
        "-i", reference_file,
        "-i", distorted_file,
        "-lavfi", f"[0:v][1:v]libvmaf='model=path={model_path}:log_fmt=json:log_path={vmaf_log}'",
        "-f", "null", "-"
    ]
    print("Running VMAF command:", " ".join(vmaf_cmd))
    subprocess.run(vmaf_cmd, check=True)

    with open(vmaf_log, "r") as f:
        data = json.load(f)

    # data["frames"] is a list of frames with "metrics": {"vmaf": XX.xx}
    vmaf_values = [float(frame["metrics"]["vmaf"]) for frame in data["frames"] if "vmaf" in frame["metrics"]]

    if vmaf_values:
        avg_vmaf = float(np.mean(vmaf_values))
        std_vmaf = float(np.std(vmaf_values))
    else:
        avg_vmaf = 0.0
        std_vmaf = 0.0

    return avg_vmaf, std_vmaf


if __name__ == "__main__":
    reference_video = "Netflix_Tango_4096x2160_60fps_10bit_420.y4m"
    encoded_video = "encoded.webm"
    chosen_bitrate = "4000k"
    
    encode_video(reference_video, output_file=encoded_video, bitrate=chosen_bitrate, encoder="libvpx")

    # Example: If you also want SSIM:
    avg_ssim, std_ssim = calculate_ssim(reference_video, encoded_video, ssim_log="ssim.log")
    print(f"[SSIM] Average: {avg_ssim:.6f}, Std: {std_ssim:.6f}")

    avg_vmaf, std_vmaf = calculate_vmaf(
        reference_video, 
        encoded_video,
        model_path="/mnt/mydata/home/goodsol/vmaf/model/vmaf_v0.6.1.json", 
        vmaf_log="vmaf.json"
    )
    print(f"[VMAF] Average: {avg_vmaf:.6f}, Std: {std_vmaf:.6f}")
