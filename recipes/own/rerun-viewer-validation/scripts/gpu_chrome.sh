#!/bin/bash
# Start Google Chrome on the RTX 5090 with full GPU rendering: GL (ANGLE on Vulkan), compositing,
# raster and WebGPU (Dawn Vulkan, NVIDIA Blackwell adapter). Headed Chrome on a private Xvfb display;
# Xvfb only receives the finished frames, every pixel is drawn by the NVIDIA GPU.
#
# usage: gpu_chrome.sh <profile_dir> [<width>x<height>] [<cdp_port>]
# prints PID=<chrome pid>, XVFB_PID=<xvfb pid>, DISPLAY=:<n>, CDP=http://127.0.0.1:<port>, LOG=<file>
# Both processes keep running. Stop them with: kill <PID> <XVFB_PID>   (never kill by name)
#
# GPU_CHROME_EXTRA_FLAGS="..." appends Chrome flags (diagnostics only).
set -euo pipefail
prof=${1:?usage: gpu_chrome.sh <profile_dir> [WxH] [port]}
size=${2:-1850x1050}
port=${3:-9333}
w=${size%x*}
h=${size#*x}
mkdir -p "$prof"
log="$prof/chrome.log"

# a free X display number of our own (never :1, Pablo's desktop)
d=120
while [ -e "/tmp/.X11-unix/X$d" ] || [ -e "/tmp/.X$d-lock" ]; do d=$((d + 1)); done
Xvfb ":$d" -screen 0 "$((w + 70))x$((h + 150))x24" -nolisten tcp >"$prof/xvfb.log" 2>&1 </dev/null &
xpid=$!
for _ in $(seq 40); do [ -e "/tmp/.X11-unix/X$d" ] && break; sleep 0.1; done

env -u WAYLAND_DISPLAY DISPLAY=":$d" \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  /opt/google/chrome/chrome \
  --ozone-platform=x11 \
  --user-data-dir="$prof" \
  --remote-debugging-address=127.0.0.1 --remote-debugging-port="$port" \
  --window-position=0,0 --window-size="$w,$h" \
  --use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan \
  --ignore-gpu-blocklist --enable-gpu-rasterization --enable-unsafe-webgpu \
  --disable-gpu-sandbox \
  --no-first-run --no-default-browser-check --disable-background-networking \
  --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows \
  ${GPU_CHROME_EXTRA_FLAGS:-} \
  about:blank >"$log" 2>&1 </dev/null &
pid=$!
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$port/json/version" >/dev/null && break; sleep 0.25; done
if ! curl -sf "http://127.0.0.1:$port/json/version" >/dev/null; then
  echo "chrome did not open CDP on port $port; log: $log" >&2
  kill "$pid" "$xpid" 2>/dev/null || true
  exit 1
fi
echo "PID=$pid"
echo "XVFB_PID=$xpid"
echo "DISPLAY=:$d"
echo "CDP=http://127.0.0.1:$port"
echo "LOG=$log"
