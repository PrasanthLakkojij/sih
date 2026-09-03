"""
Render an actual MP4 video and animated GIF of the 120s outage replay.
Uses matplotlib, PIL, and cv2/imageio to create high-quality, standalone video assets.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import imageio

TRAJ_PATH = Path('plots/phase9/demo_trajectory.json')
OUT_DIR = Path('plots/phase9')
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(TRAJ_PATH, 'r') as f:
    frames = json.load(f)

print(f"Loaded {len(frames)} trajectory frames.")

# Extract coordinates
gnss_pts = np.array([[f['gnss_lon'], f['gnss_lat']] for f in frames])
phys_pts = np.array([[f['physics_lon'], f['physics_lat']] for f in frames])
ai_pts   = np.array([[f['ai_lon'], f['ai_lat']] for f in frames])

# Bounding box with margins
all_lons = np.concatenate([gnss_pts[:, 0], phys_pts[:, 0], ai_pts[:, 0]])
all_lats = np.concatenate([gnss_pts[:, 1], phys_pts[:, 1], ai_pts[:, 1]])

lon_min, lon_max = all_lons.min() - 0.003, all_lons.max() + 0.003
lat_min, lat_max = all_lats.min() - 0.002, all_lats.max() + 0.002

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

rendered_images = []

plt.style.use('dark_background')
dpi = 100
fig_w, fig_h = 10, 6

print("Rendering video frames...")
# We decimate to ~64 keyframes or full frames for smooth 15 fps video
step = 1  # render every frame (127 frames)

for i in range(0, len(frames), step):
    f = frames[i]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    # Draw full trajectories in background (faint)
    ax.plot(gnss_pts[:, 0], gnss_pts[:, 1], color='#10b981', alpha=0.3, linewidth=2.5, linestyle=':')
    ax.plot(phys_pts[:, 0], phys_pts[:, 1], color='#ef4444', alpha=0.25, linewidth=2, linestyle=':')
    ax.plot(ai_pts[:, 0], ai_pts[:, 1], color='#3b82f6', alpha=0.3, linewidth=2.5, linestyle=':')

    # Draw traveled paths up to current frame
    ax.plot(gnss_pts[:i+1, 0], gnss_pts[:i+1, 1], color='#10b981', linewidth=3.5, label='GNSS Ground Truth')
    ax.plot(phys_pts[:i+1, 0], phys_pts[:i+1, 1], color='#ef4444', linewidth=2.5, linestyle='--', label='Physics DR (ZUPT v1)')
    ax.plot(ai_pts[:i+1, 0], ai_pts[:i+1, 1], color='#3b82f6', linewidth=3.5, label='AI-Hybrid (ZUPT + ML)')

    # Current vehicle markers
    ax.plot(f['gnss_lon'], f['gnss_lat'], 'o', color='#10b981', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.plot(f['physics_lon'], f['physics_lat'], 's', color='#ef4444', markersize=11, markeredgecolor='white', markeredgewidth=2)
    ax.plot(f['ai_lon'], f['ai_lat'], 'D', color='#3b82f6', markersize=12, markeredgecolor='white', markeredgewidth=2)

    # Start Point marker
    ax.plot(gnss_pts[0, 0], gnss_pts[0, 1], 'o', color='#fbbf24', markersize=8)
    ax.text(gnss_pts[0, 0], gnss_pts[0, 1] + 0.0006, 'Outage Start', color='#fbbf24', fontsize=9, ha='center', fontweight='bold')

    # Compute live errors
    phys_err = haversine_m(f['gnss_lat'], f['gnss_lon'], f['physics_lat'], f['physics_lon'])
    ai_err   = haversine_m(f['gnss_lat'], f['gnss_lon'], f['ai_lat'], f['ai_lon'])
    red_pct  = ((phys_err - ai_err) / max(phys_err, 1e-3)) * 100 if phys_err > 1 else 0

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°)', fontsize=10, color='#94a3b8')
    ax.set_ylabel('Latitude (°)', fontsize=10, color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, linestyle='--', color='#334155', alpha=0.7)

    # Title & Subtitle
    ax.set_title('SIH26168 Dead Reckoning Replay — Motorway GNSS Outage (S-Vw4, 120s)',
                 fontsize=12, fontweight='bold', color='#f8fafc', pad=12)

    # Floating HUD dashboard box
    hud_text = (
        f"Elapsed Time: {f['elapsed_s']:.1f} s / 126.0 s\n"
        f"Physics Error: {phys_err:.1f} m\n"
        f"AI-Hybrid Error: {ai_err:.1f} m\n"
        f"Error Reduction: {red_pct:+.1f}%"
    )
    ax.text(0.02, 0.04, hud_text, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#0f172a', edgecolor='#6366f1', alpha=0.9, linewidth=1.5))

    ax.legend(loc='upper right', framealpha=0.9, facecolor='#0f172a', edgecolor='#334155', fontsize=9)

    plt.tight_layout()

    # Convert canvas to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    rendered_images.append(img_rgb)
    plt.close(fig)

    if i % 25 == 0:
        print(f"  Frame {i}/{len(frames)} rendered...")

print("All frames rendered. Encoding MP4 video...")
mp4_path = OUT_DIR / 'phase9_outage_replay.mp4'
h, w, _ = rendered_images[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 12
out_video = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))

for img in rendered_images:
    # cv2 expects BGR
    out_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
out_video.release()
print(f"Saved MP4 video: {mp4_path.resolve()}")

# Also save an animated GIF for instant browser/PPT playback
gif_path = OUT_DIR / 'phase9_outage_replay.gif'
print("Saving animated GIF (decimated for lightweight size)...")
gif_frames = rendered_images[::2]  # 63 frames
imageio.mimsave(str(gif_path), gif_frames, fps=8)
print(f"Saved Animated GIF: {gif_path.resolve()}")
