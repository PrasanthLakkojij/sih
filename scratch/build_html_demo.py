"""
Generate standalone interactive Leaflet.js HTML map demo.
Reads plots/phase9/demo_trajectory.json and embeds it directly into phase9_map_demo.html
so that the HTML opens locally via file:// without any CORS restrictions.
"""
import json
from pathlib import Path

TRAJ_PATH = Path('plots/phase9/demo_trajectory.json')
HTML_PATH = Path('phase9_map_demo.html')

with open(TRAJ_PATH, 'r') as f:
    trajectory_data = json.load(f)

json_str = json.dumps(trajectory_data)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SIH26168 — Phase 9 Dead Reckoning Map Demo</title>
  
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">

  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: rgba(30, 41, 59, 0.95);
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --gnss: #10b981;
      --physics: #ef4444;
      --hybrid: #3b82f6;
      --accent: #6366f1;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}

    /* Top Navigation Header */
    header {{
      background: #1e293b;
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 1000;
    }}

    .header-left {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .badge {{
      background: rgba(99, 102, 241, 0.2);
      color: #818cf8;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      border: 1px solid rgba(99, 102, 241, 0.4);
      letter-spacing: 0.05em;
    }}

    h1 {{
      font-size: 1.15rem;
      font-weight: 600;
      color: #f8fafc;
    }}

    .subtitle {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-left: 8px;
    }}

    /* Main Container */
    #main-container {{
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
    }}

    #map {{
      flex: 1;
      height: 100%;
      background: #0b1329;
    }}

    /* Control Panel & Floating Metrics */
    .floating-card {{
      position: absolute;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      z-index: 1000;
      padding: 16px;
    }}

    #legend-card {{
      top: 16px;
      left: 16px;
      width: 280px;
    }}

    .legend-title {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 12px;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
      font-size: 0.85rem;
      font-weight: 500;
    }}

    .legend-line {{
      width: 24px;
      height: 4px;
      border-radius: 2px;
    }}

    .line-gnss {{
      background: var(--gnss);
    }}

    .line-physics {{
      background: repeating-linear-gradient(90deg, var(--physics) 0, var(--physics) 4px, transparent 4px, transparent 8px);
      height: 3px;
    }}

    .line-hybrid {{
      background: var(--hybrid);
    }}

    /* Metrics Dashboard */
    #metrics-card {{
      top: 16px;
      right: 16px;
      width: 320px;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 8px;
    }}

    .metric-box {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 10px;
    }}

    .metric-label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}

    .metric-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.1rem;
      font-weight: 700;
    }}

    .val-gnss {{ color: var(--gnss); }}
    .val-physics {{ color: var(--physics); }}
    .val-hybrid {{ color: var(--hybrid); }}

    .reduction-banner {{
      margin-top: 10px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 8px 12px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    /* Bottom Playback Bar */
    #playback-bar {{
      background: #1e293b;
      border-top: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      z-index: 1000;
    }}

    .slider-row {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .time-readout {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text);
      min-width: 90px;
    }}

    input[type=range] {{
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      cursor: pointer;
    }}

    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 10px rgba(99, 102, 241, 0.7);
      transition: transform 0.1s;
    }}

    input[type=range]::-webkit-slider-thumb:hover {{
      transform: scale(1.2);
    }}

    .controls-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .btn-group {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    button {{
      background: #334155;
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text);
      padding: 7px 16px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}

    button:hover {{
      background: #475569;
    }}

    button.primary {{
      background: var(--accent);
      color: #ffffff;
      border: none;
    }}

    button.primary:hover {{
      background: #4f46e5;
    }}

    .speed-select {{
      background: #334155;
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }}

    .info-tag {{
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="header-left">
      <span class="badge">SIH26168 DEMO</span>
      <h1>AI + Physics Hybrid Dead Reckoning</h1>
      <span class="subtitle">— Motorway GNSS Outage (S-Vw4, 126s)</span>
    </div>
    <div class="info-tag">
      Simulated 120s Outage • Leaflet.js • OpenStreetMap Tiles
    </div>
  </header>

  <!-- Main Map Container -->
  <div id="main-container">
    <div id="map"></div>

    <!-- Floating Legend -->
    <div class="floating-card" id="legend-card">
      <div class="legend-title">Trajectories</div>
      <div class="legend-item">
        <div class="legend-line line-gnss"></div>
        <span>GNSS Ground Truth (Actual Route)</span>
      </div>
      <div class="legend-item">
        <div class="legend-line line-hybrid"></div>
        <span>AI-Hybrid (ZUPT v1 + C1 ML)</span>
      </div>
      <div class="legend-item">
        <div class="legend-line line-physics"></div>
        <span>Physics DR (ZUPT v1 Baseline)</span>
      </div>
    </div>

    <!-- Floating Metrics Display -->
    <div class="floating-card" id="metrics-card">
      <div class="legend-title">Real-Time Accuracy Dashboard</div>
      
      <div class="metric-grid">
        <div class="metric-box">
          <div class="metric-label">Elapsed Time</div>
          <div class="metric-val" id="disp-time">0.0 s</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">True Distance</div>
          <div class="metric-val val-gnss" id="disp-dist">0 m</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Physics Error</div>
          <div class="metric-val val-physics" id="disp-phys-err">0.0 m</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">AI-Hybrid Error</div>
          <div class="metric-val val-hybrid" id="disp-ai-err">0.0 m</div>
        </div>
      </div>

      <div class="reduction-banner" id="disp-reduction">
        <span>AI Error Reduction:</span>
        <span id="reduction-pct">+0.0%</span>
      </div>
    </div>
  </div>

  <!-- Playback Control Bar -->
  <div id="playback-bar">
    <div class="slider-row">
      <span class="time-readout" id="slider-time">0.0s / 126.0s</span>
      <input type="range" id="time-slider" min="0" max="126" value="0" step="1">
    </div>

    <div class="controls-row">
      <div class="btn-group">
        <button id="btn-play" class="primary">
          <span id="play-icon">▶</span> Play
        </button>
        <button id="btn-reset">↺ Reset</button>
        <button id="btn-fit">⛶ Fit Paths</button>
      </div>

      <div class="btn-group">
        <span class="info-tag">Speed:</span>
        <select id="speed-select" class="speed-select">
          <option value="1">1x</option>
          <option value="2" selected>2x</option>
          <option value="4">4x</option>
          <option value="8">8x</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIj8ZcUDKUXe5kc5eIvvdZxxTStPf+0=" crossorigin=""></script>

  <script>
    // Embedded Trajectory Data (127 seconds at 1 Hz)
    const trajectory = {json_str};

    const maxFrame = trajectory.length - 1;
    let currentFrame = 0;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 2;

    // Initialize Map with dark/standard OSM tiles
    const startPoint = [trajectory[0].gnss_lat, trajectory[0].gnss_lon];
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView(startPoint, 14);

    // OpenStreetMap tile layer (no API key needed)
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    // Build path arrays
    const gnssPath = trajectory.map(d => [d.gnss_lat, d.gnss_lon]);
    const physicsPath = trajectory.map(d => [d.physics_lat, d.physics_lon]);
    const aiPath = trajectory.map(d => [d.ai_lat, d.ai_lon]);

    // Draw full trajectory lines (background)
    const gnssLine = L.polyline(gnssPath, {{
      color: '#10b981',
      weight: 4,
      opacity: 0.85
    }}).addTo(map);

    const physicsLine = L.polyline(physicsPath, {{
      color: '#ef4444',
      weight: 3,
      opacity: 0.8,
      dashArray: '6, 8'
    }}).addTo(map);

    const aiLine = L.polyline(aiPath, {{
      color: '#3b82f6',
      weight: 4,
      opacity: 0.95
    }}).addTo(map);

    // Start Marker
    L.circleMarker(startPoint, {{
      radius: 7,
      color: '#ffffff',
      fillColor: '#10b981',
      fillOpacity: 1,
      weight: 2
    }}).addTo(map).bindPopup('<b>Outage Start Point</b><br>GNSS Lost — Inertial DR begins');

    // End Marker for Ground Truth
    const endPoint = gnssPath[gnssPath.length - 1];
    L.circleMarker(endPoint, {{
      radius: 7,
      color: '#ffffff',
      fillColor: '#ef4444',
      fillOpacity: 1,
      weight: 2
    }}).addTo(map).bindPopup('<b>Ground Truth End Point</b> (Fix restored)');

    // Dynamic Moving Vehicle Markers
    const markerGnss = L.circleMarker(startPoint, {{
      radius: 8,
      color: '#ffffff',
      fillColor: '#10b981',
      fillOpacity: 1,
      weight: 2
    }}).addTo(map);

    const markerPhysics = L.circleMarker(startPoint, {{
      radius: 8,
      color: '#ffffff',
      fillColor: '#ef4444',
      fillOpacity: 1,
      weight: 2
    }}).addTo(map);

    const markerAi = L.circleMarker(startPoint, {{
      radius: 9,
      color: '#ffffff',
      fillColor: '#3b82f6',
      fillOpacity: 1,
      weight: 2
    }}).addTo(map);

    // Fit map bounds to all paths with comfortable padding
    function fitPaths() {{
      const allPoints = gnssPath.concat(physicsPath).concat(aiPath);
      map.fitBounds(L.latLngBounds(allPoints), {{
        padding: [60, 60],
        maxZoom: 15
      }});
    }}
    fitPaths();

    // Haversine distance in meters
    function haversineDist(lat1, lon1, lat2, lon2) {{
      const R = 6371000;
      const dLat = (lat2 - lat1) * Math.PI / 180;
      const dLon = (lon2 - lon1) * Math.PI / 180;
      const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                Math.sin(dLon/2) * Math.sin(dLon/2);
      return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }}

    // Cumulative ground truth distance
    let cumDist = [0];
    for (let i = 1; i < trajectory.length; i++) {{
      const d = haversineDist(
        trajectory[i-1].gnss_lat, trajectory[i-1].gnss_lon,
        trajectory[i].gnss_lat, trajectory[i].gnss_lon
      );
      cumDist.push(cumDist[i-1] + d);
    }}

    // Update UI Elements to Frame
    function updateFrame(frameIdx) {{
      currentFrame = Math.max(0, Math.min(frameIdx, maxFrame));
      const f = trajectory[currentFrame];

      // Move Markers
      markerGnss.setLatLng([f.gnss_lat, f.gnss_lon]);
      markerPhysics.setLatLng([f.physics_lat, f.physics_lon]);
      markerAi.setLatLng([f.ai_lat, f.ai_lon]);

      // Calculate Errors
      const physErr = haversineDist(f.gnss_lat, f.gnss_lon, f.physics_lat, f.physics_lon);
      const aiErr   = haversineDist(f.gnss_lat, f.gnss_lon, f.ai_lat, f.ai_lon);
      const dist    = cumDist[currentFrame];

      // Update Dashboard
      document.getElementById('disp-time').textContent = f.elapsed_s.toFixed(1) + ' s';
      document.getElementById('disp-dist').textContent = Math.round(dist) + ' m';
      document.getElementById('disp-phys-err').textContent = physErr.toFixed(1) + ' m';
      document.getElementById('disp-ai-err').textContent = aiErr.toFixed(1) + ' m';

      const pct = physErr > 1 ? ((physErr - aiErr) / physErr * 100) : 0;
      const pctLabel = pct >= 0 ? `+${{pct.toFixed(1)}}%` : `${{pct.toFixed(1)}}%`;
      document.getElementById('reduction-pct').textContent = pctLabel;

      // Update Slider
      document.getElementById('time-slider').value = currentFrame;
      document.getElementById('slider-time').textContent = `${{f.elapsed_s.toFixed(1)}}s / ${{trajectory[maxFrame].elapsed_s.toFixed(1)}}s`;
    }}

    // Playback Loop
    function togglePlay() {{
      if (isPlaying) {{
        pause();
      }} else {{
        play();
      }}
    }}

    function play() {{
      if (currentFrame >= maxFrame) currentFrame = 0;
      isPlaying = true;
      document.getElementById('btn-play').innerHTML = '<span>⏸</span> Pause';
      
      const intervalMs = Math.max(25, Math.round(1000 / (1 * playSpeed)));
      playInterval = setInterval(() => {{
        if (currentFrame < maxFrame) {{
          updateFrame(currentFrame + 1);
        }} else {{
          pause();
        }}
      }}, intervalMs);
    }}

    function pause() {{
      isPlaying = false;
      document.getElementById('btn-play').innerHTML = '<span>▶</span> Play';
      if (playInterval) {{
        clearInterval(playInterval);
        playInterval = null;
      }}
    }}

    // Event Listeners
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('btn-reset').addEventListener('click', () => {{
      pause();
      updateFrame(0);
    }});
    document.getElementById('btn-fit').addEventListener('click', fitPaths);

    document.getElementById('time-slider').addEventListener('input', (e) => {{
      pause();
      updateFrame(parseInt(e.target.value));
    }});

    document.getElementById('speed-select').addEventListener('change', (e) => {{
      playSpeed = parseFloat(e.target.value);
      if (isPlaying) {{
        pause();
        play();
      }}
    }});

    // Initial Render
    updateFrame(0);
  </script>
</body>
</html>
"""

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated standalone HTML map demo: {HTML_PATH.resolve()}")
