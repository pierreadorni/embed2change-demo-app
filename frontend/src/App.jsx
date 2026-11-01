import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import {
  ArrowRight,
  Activity,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Eye,
  EyeOff,
  Flag,
  LocateFixed,
  Hand,
  Layers,
  MapPin,
  Minus,
  Plus,
  RotateCcw,
  Satellite,
  ScanSearch,
  X,
} from 'lucide-react'
import zone from './data/zone.json'
import './styles.css'

const CATEGORY_META = {
  new_construction: { label: 'New construction', short: 'Construction', color: '#ff7a4a' },
  demolition_clearing: { label: 'Demolition / clearing', short: 'Cleared', color: '#ffc15b' },
  surface_change: { label: 'Surface change', short: 'Surface', color: '#e9d48a' },
  vegetation_gain: { label: 'Vegetation gain', short: 'Greened', color: '#b5df6d' },
  vegetation_loss: { label: 'Vegetation loss', short: 'Vegetation loss', color: '#8bbf72' },
  water_change: { label: 'Water change', short: 'Water', color: '#79c9c0' },
  no_visible_change: { label: 'No visible change', short: 'No change', color: '#9aa39c' },
  unclear_occluded: { label: 'Unclear / occluded', short: 'Unclear', color: '#bfaddb' },
}

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'change', label: 'Change' },
  { id: 'no_change', label: 'No change' },
]

const ZONE_WEST = 2.3815
const ZONE_SOUTH = 48.9335
const ZONE_EAST = 2.4285
const ZONE_NORTH = 48.9465
const ZONE_BOUNDS = [
  [ZONE_WEST, ZONE_SOUTH],
  [ZONE_EAST, ZONE_NORTH],
]
const DRAW_BOUNDS = [
  [2.365, 48.9245],
  [2.446, 48.9555],
]
const ZONE_CENTER = [(ZONE_WEST + ZONE_EAST) / 2, (ZONE_SOUTH + ZONE_NORTH) / 2]
const TRANSPARENT_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg=='
const DEFAULT_THRESHOLD = 0.65
const MIN_THRESHOLD = 0.1
const MAX_THRESHOLD = 1.2
const THRESHOLD_STEP = 0.05

const MAP_LAYERS = [
  { id: 'satellite', label: 'Satellite view', detail: 'Sentinel-2 imagery', icon: Satellite, needsData: false },
  { id: 'alphaearth', label: 'AlphaEarth', detail: '2017 ↔ 2024 embedding PCA', icon: Layers, needsData: true },
  { id: 'velocity', label: 'Velocity', detail: 'Embedding displacement per year', icon: Activity, needsData: true },
  { id: 'detections', label: 'Detected changes', detail: 'Thresholded at D ≥ 0.65', icon: ScanSearch, needsData: true },
]

const STAGE_DURATION = { classifying: 2800 }

const pad2 = (value) => String(value).padStart(2, '0')
const formatScore = (value) => Number(value).toFixed(2)
const metaFor = (candidate) => CATEGORY_META[candidate.category] || CATEGORY_META.no_visible_change

function statusLine(candidate) {
  if (candidate.status === 'queued') return 'Waiting in queue'
  if (candidate.status === 'limited') return 'AI analysis limited for demo'
  if (candidate.status === 'downloading') return 'Downloading imagery'
  if (candidate.status === 'analyzing') return 'Gemini is reviewing'
  if (candidate.status === 'error') return 'Analysis failed'
  return candidate.hasChange ? 'Change reported' : 'No change seen'
}

function candidateForUi(candidate) {
  return {
    ...candidate,
    title: candidate.title || `Detected area ${candidate.rank}`,
    category: candidate.has_change === false
      ? 'no_visible_change'
      : candidate.category || 'unclear_occluded',
    hasChange: candidate.has_change,
    areaHa: candidate.area_ha || 0,
    before: candidate.before_url,
    after: candidate.after_url,
    mapStatus: candidate.status === 'done' ? 'done' : candidate.status === 'error' ? 'error' : ['downloading', 'analyzing'].includes(candidate.status) ? 'reading' : 'queued',
  }
}

function markerColor(candidate) {
  if (candidate.mapStatus === 'detected') return '#f2ead8'
  if (candidate.mapStatus === 'queued') return '#b6af9c'
  if (candidate.mapStatus === 'reading') return '#d7f36b'
  if (candidate.mapStatus === 'error') return '#e8a56f'
  if (candidate.status !== 'done') return '#b6af9c'
  return metaFor(candidate).color
}

function buildCandidateFeatures(candidates) {
  return {
    type: 'FeatureCollection',
    features: candidates.map((candidate) => ({
      type: 'Feature',
      id: candidate.id,
      geometry: candidate.geometry,
      properties: {
        id: candidate.id,
        rank: candidate.rank,
        state: candidate.mapStatus || 'done',
        color: markerColor(candidate),
      },
    })),
  }
}

const MAP_STYLE = {
  version: 8,
  sources: {
    satelliteOverview: {
      type: 'raster',
      tiles: ['https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg'],
      tileSize: 256,
      minzoom: 0,
      maxzoom: 14,
    },
    pca: {
      type: 'image',
      url: TRANSPARENT_PNG,
      coordinates: [
        [ZONE_WEST, ZONE_NORTH],
        [ZONE_EAST, ZONE_NORTH],
        [ZONE_EAST, ZONE_SOUTH],
        [ZONE_WEST, ZONE_SOUTH],
      ],
    },
    velocity: { type: 'image', url: TRANSPARENT_PNG, coordinates: [[ZONE_WEST, ZONE_NORTH], [ZONE_EAST, ZONE_NORTH], [ZONE_EAST, ZONE_SOUTH], [ZONE_WEST, ZONE_SOUTH]] },
    detections: { type: 'image', url: TRANSPARENT_PNG, coordinates: [[ZONE_WEST, ZONE_NORTH], [ZONE_EAST, ZONE_NORTH], [ZONE_EAST, ZONE_SOUTH], [ZONE_WEST, ZONE_SOUTH]] },
    mask: {
      type: 'geojson',
      data: {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]],
        },
        properties: {},
      },
    },
    zone: {
      type: 'geojson',
      data: zone,
    },
    candidates: {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    },
  },
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#07100b' },
    },
    {
      id: 'satellite-overview',
      type: 'raster',
      source: 'satelliteOverview',
      paint: {
        'raster-fade-duration': 0,
        'raster-resampling': 'linear',
      },
    },
    {
      id: 'pca',
      type: 'raster',
      source: 'pca',
      paint: {
        'raster-opacity': 0,
        'raster-opacity-transition': { duration: 800, delay: 0 },
        'raster-fade-duration': 0,
        'raster-resampling': 'linear',
      },
    },
    {
      id: 'velocity',
      type: 'raster',
      source: 'velocity',
      paint: { 'raster-opacity': 0, 'raster-opacity-transition': { duration: 220 }, 'raster-fade-duration': 0, 'raster-resampling': 'linear' },
    },
    {
      id: 'detections',
      type: 'raster',
      source: 'detections',
      paint: { 'raster-opacity': 0, 'raster-opacity-transition': { duration: 220 }, 'raster-fade-duration': 0, 'raster-resampling': 'nearest' },
    },
    {
      id: 'zone-fill',
      type: 'fill',
      source: 'zone',
      paint: {
        'fill-color': '#e9e4d2',
        'fill-opacity': 0,
        'fill-opacity-transition': { duration: 600, delay: 0 },
      },
    },
    {
      id: 'candidate-fill',
      type: 'fill',
      source: 'candidates',
      paint: {
        'fill-color': ['get', 'color'],
        'fill-opacity': [
          'case',
          ['==', ['get', 'state'], 'detected'], 0.07,
          ['==', ['get', 'state'], 'queued'], 0.05,
          ['==', ['get', 'state'], 'reading'], 0.22,
          0.19,
        ],
      },
    },
    {
      id: 'candidate-line',
      type: 'line',
      source: 'candidates',
      paint: {
        'line-color': ['get', 'color'],
        'line-width': [
          'case',
          ['==', ['get', 'state'], 'detected'], 1.15,
          ['==', ['get', 'state'], 'queued'], 0.9,
          1.7,
        ],
        'line-opacity': [
          'case',
          ['==', ['get', 'state'], 'queued'], 0.5,
          ['==', ['get', 'state'], 'detected'], 0.72,
          0.92,
        ],
      },
    },
    {
      id: 'candidate-selected-fill',
      type: 'fill',
      source: 'candidates',
      filter: ['==', ['get', 'id'], 'none'],
      paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.4 },
    },
    {
      id: 'candidate-selected-line',
      type: 'line',
      source: 'candidates',
      filter: ['==', ['get', 'id'], 'none'],
      paint: {
        'line-color': '#fff9e9',
        'line-width': 2.4,
        'line-blur': 0.5,
        'line-opacity': 1,
      },
    },
  ],
}

function mapPadding(mode) {
  const width = window.innerWidth
  const height = window.innerHeight
  if (width <= 900) {
    if (mode === 'results') return { top: 86, right: 18, bottom: Math.round(height * 0.62), left: 18 }
    return { top: 86, right: 24, bottom: 300, left: 24 }
  }
  if (mode === 'results') {
    return { top: 116, right: 490, bottom: 268, left: 76 }
  }
  if (mode === 'draw') {
    return { top: 106, right: 92, bottom: 250, left: 92 }
  }
  return { top: 112, right: 92, bottom: 246, left: 92 }
}

function LayerSelector({ activeLayer, onSelect, analysis, comparePosition, onCompareChange, onComparePointerDown, onPanelPointerEnter, onPanelPointerLeave, threshold, onThresholdChange, thresholdUpdating }) {
  const hasData = Boolean(analysis)
  const detectedCount = analysis?.detected_pixels?.toLocaleString() || '—'
  const detectedComponents = analysis?.detected_components?.toLocaleString() || '—'

  return (
    <section
      className="layer-selector"
      aria-label="Map layers"
      onPointerEnter={onPanelPointerEnter}
      onPointerLeave={onPanelPointerLeave}
    >
      <div className="layer-selector-head">
        <div>
          <h2>Map layers</h2>
          <p>{hasData ? '2017 → 2024 · live analysis' : 'Choose a view'}</p>
        </div>
        <span className={`layer-ready ${hasData ? 'is-ready' : ''}`} aria-label={hasData ? 'Analysis layers ready' : 'Analysis layers need a scan'} />
      </div>
      <div className="layer-options" role="group" aria-label="Choose a map layer">
        {MAP_LAYERS.map(({ id, label, detail, icon: Icon, needsData }) => {
          const layerDetail = id === 'detections' && hasData ? `D ≥ ${threshold.toFixed(2)}` : detail
          return (
            <button
              type="button"
              key={id}
              aria-pressed={activeLayer === id}
              aria-label={`${label}${needsData && !hasData ? ', available after running a scan' : ''}`}
              className={`layer-option ${activeLayer === id ? 'is-active' : ''}`}
              disabled={needsData && !hasData}
              onClick={() => onSelect(id)}
            >
              <span className="layer-option-icon"><Icon size={16} strokeWidth={1.8} /></span>
              <span className="layer-option-copy"><strong>{label}</strong><small>{layerDetail}</small></span>
              <span className="layer-option-state" aria-hidden="true" />
            </button>
          )
        })}
      </div>

      {activeLayer === 'alphaearth' && hasData && (
        <div className="layer-detail alphaearth-detail">
          <div className="layer-year-labels"><span>2017</span><span>2024</span></div>
          <input
            type="range"
            min="0"
            max="100"
            value={comparePosition}
            onChange={(event) => onCompareChange(Number(event.target.value))}
            onPointerDown={onComparePointerDown}
            onMouseDown={(event) => event.stopPropagation()}
            onTouchStart={(event) => event.stopPropagation()}
            aria-label="Reveal the 2017 AlphaEarth layer"
            style={{ '--range-progress': `${comparePosition}%` }}
          />
          <p>Drag the map divider or this slider to compare the years.</p>
        </div>
      )}

      {activeLayer === 'velocity' && hasData && (
        <div className="layer-detail velocity-detail">
          <div className="velocity-legend" />
          <div className="velocity-scale"><span>0</span><span>0.14</span><span>0.29 D/yr</span></div>
        </div>
      )}

      {activeLayer === 'detections' && hasData && (
        <div className="layer-detail detection-detail">
          <div className="detection-summary">
            <span className="detection-swatch" />
            <span><strong>{detectedCount}</strong> pixels retained</span>
            <small>{detectedComponents} connected areas · &gt;50 px · ≈ {Number.isFinite(analysis.detected_area_ha) ? analysis.detected_area_ha.toFixed(1) : '—'} ha</small>
          </div>
          <div className="threshold-control">
            <label htmlFor="detection-threshold">
              <span>Detection threshold</span>
              <strong>D ≥ {threshold.toFixed(2)}</strong>
            </label>
            <input
              id="detection-threshold"
              type="range"
              min={MIN_THRESHOLD}
              max={MAX_THRESHOLD}
              step={THRESHOLD_STEP}
              value={threshold}
              onChange={(event) => onThresholdChange(Number(event.target.value))}
              aria-label="Minimum total embedding displacement for detection"
            />
            <div className="threshold-endpoints"><span>More detections</span><span>Fewer detections</span></div>
            <p aria-live="polite">{thresholdUpdating ? 'Updating detection mask…' : `${detectedCount} pixels in retained areas`}</p>
          </div>
        </div>
      )}

      {!hasData && <p className="layer-empty">Run a scan to load embedding, velocity, and detection layers.</p>}
    </section>
  )
}

function MapStage({ stage, candidates, selectedId, onSelect, onDrawComplete, scanEpoch, analysis, selectedBounds, comparisonImage, activeLayer, onLayerSelect, comparePosition, onCompareChange, threshold, onThresholdChange, thresholdUpdating }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const markerById = useRef(new Map())
  const stageRef = useRef(stage)
  const onSelectRef = useRef(onSelect)
  const onDrawCompleteRef = useRef(onDrawComplete)
  const drawStateRef = useRef({ active: false, start: null, box: null })
  const syncZoneRef = useRef(() => {})
  const [mapReady, setMapReady] = useState(false)
  const [drawBox, setDrawBox] = useState(null)
  const [zoneRect, setZoneRect] = useState(null)
  const [overlaysVisible, setOverlaysVisible] = useState(true)
  const [panMode, setPanMode] = useState(false)
  const panModeRef = useRef(false)
  const comparisonDragRef = useRef(false)
  const layerPanelHoverRef = useRef(false)

  const visibleCandidates = useMemo(() => activeLayer === 'satellite' ? candidates : [], [activeLayer, candidates])
  const candidateIds = visibleCandidates.map((candidate) => candidate.id).join(',')

  useEffect(() => {
    stageRef.current = stage
  }, [stage])

  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])

  useEffect(() => {
    onDrawCompleteRef.current = onDrawComplete
  }, [onDrawComplete])

  useEffect(() => {
    const finishComparisonDrag = () => {
      if (!comparisonDragRef.current) return
      comparisonDragRef.current = false
      if (stageRef.current !== 'draw' && !layerPanelHoverRef.current) mapRef.current?.dragPan.enable()
    }
    window.addEventListener('pointerup', finishComparisonDrag)
    window.addEventListener('pointercancel', finishComparisonDrag)
    window.addEventListener('mouseup', finishComparisonDrag)
    window.addEventListener('touchend', finishComparisonDrag)
    return () => {
      window.removeEventListener('pointerup', finishComparisonDrag)
      window.removeEventListener('pointercancel', finishComparisonDrag)
      window.removeEventListener('mouseup', finishComparisonDrag)
      window.removeEventListener('touchend', finishComparisonDrag)
    }
  }, [])

  syncZoneRef.current = () => {
    const map = mapRef.current
    if (!map) return
    const [[west, south], [east, north]] = selectedBounds
    const northWest = map.project([west, north])
    const southEast = map.project([east, south])
    setZoneRect({
      left: Math.min(northWest.x, southEast.x),
      top: Math.min(northWest.y, southEast.y),
      width: Math.abs(southEast.x - northWest.x),
      height: Math.abs(southEast.y - northWest.y),
    })
  }

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return undefined
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: ZONE_CENTER,
      zoom: 13.4,
      minZoom: 8,
      maxZoom: 18,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
      bearing: 0,
      pitch: 0,
      antialias: true,
    })
    mapRef.current = map

    map.on('load', () => {
      setMapReady(true)
      map.on('move', () => syncZoneRef.current())
      ;['candidate-fill', 'candidate-line'].forEach((layer) => {
        map.on('click', layer, (event) => {
          const feature = event.features?.[0]
          const id = feature?.properties?.id
          if (!['done', 'error'].includes(feature?.properties?.state)) return
          if (!id) return
          event.stopPropagation()
          onSelectRef.current(id)
        })
        map.on('mouseenter', layer, (event) => {
          if (['done', 'error'].includes(event.features?.[0]?.properties?.state)) map.getCanvas().style.cursor = 'pointer'
        })
        map.on('mouseleave', layer, () => {
          map.getCanvas().style.cursor = stageRef.current === 'draw' ? 'crosshair' : ''
        })
      })
    })

    const pointOf = (event) => event.point || event.points?.[0] || null

    const startDraw = (event) => {
      if (stageRef.current !== 'draw' || panModeRef.current) return
      if (event.originalEvent?.button != null && event.originalEvent.button !== 0) return
      if (event.points && event.points.length > 1) return
      const point = pointOf(event)
      if (!point) return
      drawStateRef.current = {
        active: true,
        start: { x: point.x, y: point.y },
        box: { left: point.x, top: point.y, width: 0, height: 0 },
      }
      setDrawBox(drawStateRef.current.box)
    }

    const moveDraw = (event) => {
      const draw = drawStateRef.current
      if (!draw.active) return
      const point = pointOf(event)
      if (!point) return
      const left = Math.min(draw.start.x, point.x)
      const top = Math.min(draw.start.y, point.y)
      const box = {
        left,
        top,
        width: Math.abs(point.x - draw.start.x),
        height: Math.abs(point.y - draw.start.y),
      }
      draw.box = box
      setDrawBox(box)
    }

    const endDraw = () => {
      const draw = drawStateRef.current
      if (!draw.active) return
      draw.active = false
      const box = draw.box
      draw.box = null
      setDrawBox(null)
      if (box && box.width >= 32 && box.height >= 24) {
        const northWest = map.unproject([box.left, box.top])
        const southEast = map.unproject([box.left + box.width, box.top + box.height])
        onDrawCompleteRef.current?.([[northWest.lng, southEast.lat], [southEast.lng, northWest.lat]])
      }
    }

    map.on('mousedown', startDraw)
    map.on('mousemove', moveDraw)
    map.on('mouseup', endDraw)
    map.on('touchstart', startDraw)
    map.on('touchmove', moveDraw)
    map.on('touchend', endDraw)
    window.addEventListener('mouseup', endDraw)

    const resizeObserver = new ResizeObserver(() => map.resize())
    resizeObserver.observe(containerRef.current)

    return () => {
      window.removeEventListener('mouseup', endDraw)
      resizeObserver.disconnect()
      markerById.current.forEach((marker) => marker.remove())
      markerById.current.clear()
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    if (stage === 'draw') {
      if (panMode) map.dragPan.enable()
      else map.dragPan.disable()
      map.boxZoom.disable()
      map.getCanvas().style.cursor = panMode ? 'grab' : 'crosshair'
    } else {
      if (layerPanelHoverRef.current || comparisonDragRef.current) map.dragPan.disable()
      else map.dragPan.enable()
      map.boxZoom.enable()
      map.getCanvas().style.cursor = ''
    }
    if (stage !== 'draw') {
      drawStateRef.current.active = false
      drawStateRef.current.box = null
      setDrawBox(null)
    }
  }, [stage, mapReady, panMode])

  useEffect(() => {
    panModeRef.current = panMode
  }, [panMode])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    if (stage === 'draw') {
      map.fitBounds(DRAW_BOUNDS, { padding: mapPadding('draw'), duration: 900, essential: true })
      return
    }
    if (stage === 'loading') {
      map.fitBounds(selectedBounds, { padding: mapPadding('process'), duration: 1200, essential: true })
      return
    }
    if (stage === 'results') {
      map.fitBounds(selectedBounds, { padding: mapPadding('results'), duration: 1000, essential: true })
    }
  }, [stage, mapReady, scanEpoch, selectedBounds])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    const source = map.getSource('candidates')
    if (source) source.setData(buildCandidateFeatures(visibleCandidates))
  }, [visibleCandidates, mapReady])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    const filter = ['==', ['get', 'id'], selectedId || 'none']
    ;['candidate-selected-fill', 'candidate-selected-line'].forEach((layer) => {
      if (map.getLayer(layer)) map.setFilter(layer, filter)
    })
  }, [selectedId, mapReady])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    const present = new Set(visibleCandidates.map((candidate) => candidate.id))
    markerById.current.forEach((marker, id) => {
      if (!present.has(id)) {
        marker.remove()
        markerById.current.delete(id)
      }
    })
    visibleCandidates.forEach((candidate, index) => {
      if (markerById.current.has(candidate.id)) return
      const element = document.createElement('button')
      element.type = 'button'
      element.className = `candidate-marker ${candidate.mapStatus || 'done'}`
      element.disabled = !['done', 'error'].includes(candidate.mapStatus)
      element.style.setProperty('--marker-color', markerColor(candidate))
      element.style.setProperty('--marker-delay', `${Math.min(index * 34, 300)}ms`)
      element.innerHTML = `<span class="marker-disc"><span class="marker-rank">${pad2(candidate.rank)}</span></span>`
      element.addEventListener('click', (event) => {
        event.stopPropagation()
        onSelectRef.current(candidate.id)
      })
      const marker = new maplibregl.Marker({ element, anchor: 'center' })
        .setLngLat(candidate.centroid)
        .addTo(map)
      markerById.current.set(candidate.id, marker)
    })
  }, [candidateIds, mapReady])

  useEffect(() => {
    visibleCandidates.forEach((candidate) => {
      const marker = markerById.current.get(candidate.id)
      if (!marker) return
      const element = marker.getElement()
      const selected = candidate.id === selectedId
      element.className = `candidate-marker ${candidate.mapStatus || 'done'} ${selected ? 'is-selected' : ''} ${overlaysVisible ? '' : 'markers-hidden'}`
      element.disabled = !['done', 'error'].includes(candidate.mapStatus)
      element.style.setProperty('--marker-color', markerColor(candidate))
      element.setAttribute('aria-pressed', String(selected))
      const accessible = candidate.mapStatus === 'reading'
        ? `${candidate.id}, reading the paired imagery`
        : candidate.mapStatus === 'queued'
          ? `${candidate.id}, queued for reading`
          : `${candidate.id}, ${candidate.title}, ${statusLine(candidate)}`
      element.setAttribute('aria-label', accessible)
      element.title = accessible
    })
  }, [visibleCandidates, selectedId, overlaysVisible, stage])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    const [[west, south], [east, north]] = selectedBounds
    map.getSource('zone')?.setData({
      type: 'Feature',
      properties: {},
      geometry: { type: 'Polygon', coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]] },
    })
  }, [selectedBounds, mapReady])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady) return
    const pcaOpacity = activeLayer === 'alphaearth' ? 0.93 : 0
    const velocityOpacity = activeLayer === 'velocity' ? 0.9 : 0
    const detectionsOpacity = activeLayer === 'detections' ? 0.94 : 0
    const zoneOpacity = ['loading', 'classifying', 'detected', 'interpreting'].includes(stage) ? 0.035 : 0
    map.setPaintProperty('pca', 'raster-opacity', pcaOpacity)
    map.setPaintProperty('velocity', 'raster-opacity', velocityOpacity)
    map.setPaintProperty('detections', 'raster-opacity', detectionsOpacity)
    map.setPaintProperty('zone-fill', 'fill-opacity', zoneOpacity)
    if (stage === 'draw') {
      setOverlaysVisible(true)
      ;['candidate-fill', 'candidate-line', 'candidate-selected-fill', 'candidate-selected-line'].forEach((layer) => {
        if (map.getLayer(layer)) map.setLayoutProperty(layer, 'visibility', 'visible')
      })
    }
  }, [stage, mapReady, activeLayer])

  useEffect(() => {
    if (!comparisonImage || !mapReady) return
    const [[west, south], [east, north]] = selectedBounds
    const coordinates = [[west, north], [east, north], [east, south], [west, south]]
    mapRef.current?.getSource('pca')?.updateImage({ url: comparisonImage, coordinates })
  }, [comparisonImage, mapReady, selectedBounds])

  useEffect(() => {
    if (!analysis?.velocity || !mapReady) return
    const [[west, south], [east, north]] = selectedBounds
    const coordinates = [[west, north], [east, north], [east, south], [west, south]]
    mapRef.current?.getSource('velocity')?.updateImage({ url: `data:image/png;base64,${analysis.velocity}`, coordinates })
  }, [analysis?.velocity, mapReady, selectedBounds])

  useEffect(() => {
    if (!analysis?.detections || !mapReady) return
    const [[west, south], [east, north]] = selectedBounds
    const coordinates = [[west, north], [east, north], [east, south], [west, south]]
    mapRef.current?.getSource('detections')?.updateImage({ url: `data:image/png;base64,${analysis.detections}`, coordinates })
  }, [analysis?.detections, mapReady, selectedBounds])

  useEffect(() => {
    if (mapReady) syncZoneRef.current()
  }, [stage, mapReady])

  const zoom = (direction) => {
    const map = mapRef.current
    if (!map) return
    if (direction === 'in') map.zoomIn({ duration: 350 })
    else map.zoomOut({ duration: 350 })
  }

  const togglePanMode = () => setPanMode((current) => !current)

  const beginComparisonDrag = (event) => {
    event.stopPropagation()
    comparisonDragRef.current = true
    mapRef.current?.dragPan.disable()
  }

  const enterLayerPanel = () => {
    layerPanelHoverRef.current = true
    mapRef.current?.dragPan.disable()
  }

  const leaveLayerPanel = () => {
    layerPanelHoverRef.current = false
    if (!comparisonDragRef.current && stageRef.current !== 'draw') mapRef.current?.dragPan.enable()
  }

  const setComparisonFromPointer = (event) => {
    const canvasBounds = containerRef.current?.getBoundingClientRect()
    if (!canvasBounds || !zoneRect?.width) return
    const position = ((event.clientX - canvasBounds.left - zoneRect.left) / zoneRect.width) * 100
    onCompareChange(Math.max(0, Math.min(100, Math.round(position))))
  }

  const beginMapComparisonDrag = (event) => {
    event.preventDefault()
    beginComparisonDrag(event)
    event.currentTarget.setPointerCapture?.(event.pointerId)
    setComparisonFromPointer(event)
  }

  const moveMapComparisonDrag = (event) => {
    if (comparisonDragRef.current) setComparisonFromPointer(event)
  }

  const keyMapComparison = (event) => {
    const increments = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 }
    if (event.key === 'Home') onCompareChange(0)
    else if (event.key === 'End') onCompareChange(100)
    else if (increments[event.key]) onCompareChange(Math.max(0, Math.min(100, comparePosition + increments[event.key])))
    else return
    event.preventDefault()
  }

  const fitView = () => {
    const map = mapRef.current
    if (!map) return
    if (stageRef.current === 'draw') {
      map.fitBounds(DRAW_BOUNDS, { padding: mapPadding('draw'), duration: 900, essential: true })
      return
    }
    map.fitBounds(ZONE_BOUNDS, {
      padding: mapPadding(stageRef.current === 'results' ? 'results' : 'process'),
      duration: 900,
      essential: true,
    })
  }

  const toggleOverlays = () => {
    const map = mapRef.current
    if (!map) return
    const next = !overlaysVisible
    setOverlaysVisible(next)
    ;['candidate-fill', 'candidate-line', 'candidate-selected-fill', 'candidate-selected-line'].forEach((layer) => {
      if (map.getLayer(layer)) map.setLayoutProperty(layer, 'visibility', next ? 'visible' : 'none')
    })
    markerById.current.forEach((marker) => {
      marker.getElement().classList.toggle('markers-hidden', !next)
    })
  }

  const zoneVisible = zoneRect && ['loading', 'classifying', 'detected'].includes(stage)

  return (
    <div className="map-stage" data-stage={stage} data-ready={mapReady ? 'true' : 'false'}>
      <div ref={containerRef} className="map-canvas" role="region" aria-label="Map and selected analysis zone" />
      <LayerSelector
        activeLayer={activeLayer}
        onSelect={onLayerSelect}
        analysis={analysis}
        comparePosition={comparePosition}
        onCompareChange={onCompareChange}
        onComparePointerDown={beginComparisonDrag}
        onPanelPointerEnter={enterLayerPanel}
        onPanelPointerLeave={leaveLayerPanel}
        threshold={threshold}
        onThresholdChange={onThresholdChange}
        thresholdUpdating={thresholdUpdating}
      />

      {activeLayer === 'alphaearth' && zoneRect && (
        <div
          className="alphaearth-divider"
          role="slider"
          tabIndex={0}
          aria-label="Swipe between AlphaEarth 2017 and 2024"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={comparePosition}
          aria-valuetext={`${comparePosition}% of the map shows 2017`}
          onPointerDown={beginMapComparisonDrag}
          onPointerMove={moveMapComparisonDrag}
          onKeyDown={keyMapComparison}
          style={{ left: `${zoneRect.left + zoneRect.width * comparePosition / 100}px`, top: `${zoneRect.top}px`, height: `${zoneRect.height}px` }}
        >
          <span className="alphaearth-divider-handle"><ChevronLeft size={13} /><ChevronRight size={13} /></span>
          <span className="alphaearth-map-label alphaearth-map-label--2017">2017</span>
          <span className="alphaearth-map-label alphaearth-map-label--2024">2024</span>
        </div>
      )}

      {zoneVisible && (
        <div
          className={`zone-frame is-${stage}`}
          style={{
            left: `${zoneRect.left}px`,
            top: `${zoneRect.top}px`,
            width: `${zoneRect.width}px`,
            height: `${zoneRect.height}px`,
          }}
          aria-hidden="true"
        >
          <span className="zone-corner zone-corner--tl" />
          <span className="zone-corner zone-corner--tr" />
          <span className="zone-corner zone-corner--bl" />
          <span className="zone-corner zone-corner--br" />
          <span className="zone-scan"><span /></span>
        </div>
      )}

      {drawBox && (
        <div
          className="draw-box"
          style={{
            left: `${drawBox.left}px`,
            top: `${drawBox.top}px`,
            width: `${drawBox.width}px`,
            height: `${drawBox.height}px`,
          }}
          aria-hidden="true"
        />
      )}

      <div className="map-attribution">
        Imagery © <a href="https://cloudless.eox.at" target="_blank" rel="noreferrer">EOxCloudless</a> · Modified Copernicus Sentinel data 2020
      </div>

      <div className="map-controls" aria-label="Map controls">
        {stage === 'draw' && (
          <button
            type="button"
            onClick={togglePanMode}
            aria-label={panMode ? 'Switch to zone selection' : 'Switch to map pan'}
            aria-pressed={panMode}
            title={panMode ? 'Switch to zone selection' : 'Switch to map pan'}
            className={panMode ? 'is-active' : ''}
          >
            <Hand size={16} strokeWidth={1.8} />
          </button>
        )}
        <button type="button" onClick={() => zoom('in')} aria-label="Zoom in" title="Zoom in">
          <Plus size={17} strokeWidth={1.8} />
        </button>
        <button type="button" onClick={() => zoom('out')} aria-label="Zoom out" title="Zoom out">
          <Minus size={17} strokeWidth={1.8} />
        </button>
        <span className="control-rule" />
        <button type="button" onClick={fitView} aria-label="Fit the analysis view" title="Fit the analysis view">
          <LocateFixed size={16} strokeWidth={1.8} />
        </button>
        {stage === 'results' && (
          <>
            <span className="control-rule" />
            <button
              type="button"
              onClick={toggleOverlays}
              aria-label={overlaysVisible ? 'Hide change overlays' : 'Show change overlays'}
              title={overlaysVisible ? 'Hide change overlays' : 'Show change overlays'}
              className={!overlaysVisible ? 'is-muted' : ''}
            >
              {overlaysVisible ? <Eye size={16} strokeWidth={1.8} /> : <EyeOff size={16} strokeWidth={1.8} />}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function CompareViewer({ candidate }) {
  const [split, setSplit] = useState(52)
  const [imageError, setImageError] = useState(false)

  useEffect(() => {
    setSplit(52)
    setImageError(false)
  }, [candidate.id])

  return (
    <figure className="compare-block" aria-label={`Before and after orthophotos for candidate ${candidate.id}`}>
      <div className="compare-frame" style={{ '--split': `${split}%` }}>
        {candidate.after && <img className="compare-image compare-after" src={candidate.after} alt={`${candidate.id} outlined after imagery, 2024`} onError={() => setImageError(true)} draggable="false" />}
        {candidate.before && <img
          className="compare-image compare-before"
          src={candidate.before}
          alt={`${candidate.id} outlined before imagery, 2017`}
          draggable="false"
          style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
          onError={() => setImageError(true)}
        />}
        <span className="compare-label compare-label--before">2017 <b>before</b></span>
        <span className="compare-label compare-label--after">2024 <b>after</b></span>
        <div className="compare-divider" aria-hidden="true">
          <span className="compare-handle">
            <ChevronLeft size={13} strokeWidth={2} />
            <ChevronRight size={13} strokeWidth={2} />
          </span>
        </div>
        <input
          className="compare-range"
          type="range"
          min="0"
          max="100"
          value={split}
          onChange={(event) => setSplit(Number(event.target.value))}
          aria-label="Reveal 2017 and 2024 orthophotos"
        />
        {(imageError || !candidate.before || !candidate.after) && <span className="compare-error" role="status">Orthophoto preview unavailable. {candidate.error || 'The VLM review may still be complete.'}</span>}
      </div>
      <figcaption>Aligned IGN orthophotos</figcaption>
    </figure>
  )
}

function CandidateCard({ candidate, selected, onSelect, cardRef }) {
  const meta = metaFor(candidate)
  return (
    <button
      type="button"
      ref={cardRef}
      className={`queue-card ${selected ? 'is-selected' : ''}`}
      onClick={() => onSelect(candidate.id)}
      aria-pressed={selected}
      aria-label={`Candidate ${candidate.id}, rank ${pad2(candidate.rank)}, ${meta.label}, ${statusLine(candidate)}`}
    >
      <span className={`queue-thumb ${candidate.status === 'limited' ? 'queue-thumb--limited' : ''}`}>
        {candidate.after
          ? <img src={candidate.after} alt={`${candidate.title}, 2024`} loading="lazy" draggable="false" />
          : <span>{candidate.status === 'limited' ? 'AI limit' : 'Pending'}</span>}
      </span>
      <span className="queue-copy">
        <span className="queue-kicker">
          <span className="category-dot" style={{ background: meta.color }} />
          {meta.short}
        </span>
        <span className="queue-title">{candidate.title}</span>
        <span className="queue-meta">
          <span>#{pad2(candidate.rank)}</span>
          <span className="queue-score">D {formatScore(candidate.score)}</span>
        </span>
      </span>
    </button>
  )
}

function MethodPanel({ onClose }) {
  return (
    <section className="method-panel" aria-label="How this works">
      <div className="method-head">
        <h2>How this works</h2>
        <button type="button" className="icon-button icon-button--small" onClick={onClose} aria-label="Close method panel">
          <X size={16} strokeWidth={1.8} />
        </button>
      </div>
      <p className="method-intro">
        Select a zone, compare two years of AlphaEarth embeddings,
        then let a vision model read the flagged ground.
      </p>
      <ol className="method-list">
        <li>
          <div>
            <h3>Select a zone</h3>
            <p>Draw a rectangle anywhere on the map to analyze that area.</p>
          </div>
        </li>
        <li>
          <div>
            <h3>Compare embeddings</h3>
            <p>Real 2017 and 2024 AlphaEarth vectors share a three-component PCA space. Total displacement is measured for every valid pixel.</p>
          </div>
        </li>
        <li>
          <div>
            <h3>Detect candidates</h3>
            <p>Pixels with total displacement D&nbsp;≥&nbsp;0.65 are highlighted as detected change. D is a screening signal, not a probability.</p>
          </div>
        </li>
        <li>
          <div>
            <h3>Read the ground</h3>
            <p>Aligned IGN before/after crops are outlined and reviewed one at a time by Gemini. Treat results as a screening aid.</p>
          </div>
        </li>
      </ol>
      <p className="method-note">
        Embeddings, connected regions, and IGN imagery are processed for the selected area; Gemini provides hosted reviews.
      </p>
    </section>
  )
}

function DrawDock() {
  return (
    <section className="draw-dock" aria-label="Select an analysis zone">
      <div className="draw-copy">
        <p className="dock-eyebrow">
          <MapPin size={14} strokeWidth={1.8} />
          2017 → 2024
        </p>
        <h2>Draw an area to compare</h2>
        <p>
          Drag a rectangle on the map to choose the area for analysis.
        </p>
      </div>
    </section>
  )
}

function BatchGrid({ candidates, reads }) {
  return (
    <div className="batch-grid" role="list" aria-label="Candidate reading progress">
      {candidates.map((candidate) => {
        const state = reads[candidate.id] || 'queued'
        const meta = metaFor(candidate)
        return (
          <span
            role="listitem"
            key={candidate.id}
            className={`batch-cell is-${state}`}
            style={{ '--cell-color': meta.color }}
            title={
              state === 'reading'
                ? `${candidate.id}, reading`
                : state === 'done'
                  ? `${candidate.id}, ${meta.label}`
                  : state === 'error'
                    ? `${candidate.id}, analysis failed`
                  : `${candidate.id}, queued`
            }
            aria-label={
              state === 'reading'
                ? `${candidate.id}, reading`
                : state === 'done'
                  ? `${candidate.id}, ${meta.label}`
                  : state === 'error'
                    ? `${candidate.id}, analysis failed`
                  : `${candidate.id}, queued`
            }
          >
            {pad2(candidate.rank)}
          </span>
        )
      })}
    </div>
  )
}

const PROCESS_COPY = {
  loading: {
    eyebrow: 'AlphaEarth · 2017 → 2024',
    title: 'Loading embeddings',
    body: 'Downloading real 2017 and 2024 embeddings for the 4.98\u00a0km² zone.',
  },
  classifying: {
    eyebrow: '3-component PCA · AlphaEarth',
    title: 'Projecting the change field',
    body: 'The paired 64-band embeddings are projected into three principal components.',
  },
  detected: {
    eyebrow: 'Detector · D ≥ 0.65',
    title: 'Change regions are ready',
    body: 'Review the thresholded regions, then send them to the hosted vision model.',
  },
  interpreting: {
    eyebrow: 'Gemini · hosted review',
    title: 'Review queue is running',
    body: 'The worker downloads paired IGN imagery, outlines each region, and reviews the composite.',
  },
}

function ProcessDock({ stage, completed, total, readingIds, candidates, onCompute, busy, analysis, reviewJob }) {
  const copy = stage === 'detected' && analysis
    ? { ...PROCESS_COPY.detected, title: `${analysis.detected_pixels.toLocaleString()} pixels in ${analysis.detected_components.toLocaleString()} areas`, body: `About ${analysis.detected_area_ha.toFixed(1)} ha remain at D ≥ ${analysis.threshold.toFixed(2)} after the >50 px area filter.` }
    : PROCESS_COPY[stage]
  const interpreting = stage === 'interpreting'
  const progress = interpreting ? (total ? Math.round((completed / total) * 100) : 100) : stage === 'detected' ? 100 : 0

  return (
    <section className={`process-dock stage-${stage}`} data-stage={stage} aria-label={copy.title}>
      <div className="process-main">
        <div className="process-copy">
          <p className="dock-eyebrow">{copy.eyebrow}</p>
          <h2>{copy.title}</h2>
          <p className="process-body">{copy.body}</p>
          {stage === 'classifying' && (
            <p className="pca-key">
              <span className="pca-swatch" style={{ background: '#297967' }} />
              <span className="pca-swatch" style={{ background: '#cccc74' }} />
              <span className="pca-swatch" style={{ background: '#e87452' }} />
              3-component PCA field
            </p>
          )}
          {stage === 'detected' && (
            <p className="process-footnote">AI-assisted analysis is limited to 15 changes per selected zone in this prototype. Other detected zones remain visible on the map.</p>
          )}
          {interpreting && (
            <p className="process-count" aria-live="polite">
              {total === 0 ? 'No connected regions exceed the minimum area.' : `${completed} of ${total} reviewed`}
              <span>{readingIds.length ? ` · ${readingIds[0]} ${reviewJob?.candidates?.find((candidate) => candidate.id === readingIds[0])?.status || 'working'}` : ''}</span>
            </p>
          )}
        </div>

        {stage === 'detected' && (
          <div className="batch-read">
            <p className="provider-detail">Hosted Gemini review requires a Google AI Studio API key.</p>
            <button type="button" className="primary-action" onClick={onCompute} disabled={busy || !analysis?.detected_components}>
              <ScanSearch size={16} />
              {busy ? 'Preparing queue…' : 'Compute VLM analysis'}
            </button>
          </div>
        )}

        {interpreting && (
          <div className="batch-read">
            <BatchGrid candidates={candidates} reads={Object.fromEntries(candidates.map((candidate) => [candidate.id, candidate.mapStatus]))} />
            {reviewJob?.candidates?.some((candidate) => candidate.analysis_eligible === false) && (
              <p className="process-footnote">AI-assisted analysis is limited to 15 changes per selected zone in this prototype. Other detected zones remain visible on the map.</p>
            )}
          </div>
        )}
      </div>

      <div
        className="process-progress"
        role="progressbar"
        aria-label={interpreting ? 'Candidates read' : copy.title}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={interpreting || stage === 'detected' ? progress : undefined}
      >
        <span
          className={interpreting ? 'is-live' : ''}
          style={interpreting || stage === 'detected'
            ? { transform: `scaleX(${progress / 100})` }
          : { '--duration': `${STAGE_DURATION[stage] || 0}ms` }}
        />
      </div>

    </section>
  )
}

function ResultsDock({
  candidates,
  filtered,
  selectedId,
  filter,
  counts,
  onFilter,
  onSelect,
  onScroll,
  queueRef,
  cardRefs,
  reviewJob,
}) {
  return (
    <section className="results-dock" aria-label="Results queue">
      <div className="results-head">
        <div className="results-title-wrap">
          <h2>Results</h2>
          <p>{reviewJob?.total ?? candidates.length} regions analyzed · {counts.change} change · {counts.no_change} no change</p>
        </div>

        {reviewJob?.candidates?.some((candidate) => candidate.analysis_eligible === false) && (
          <p className="results-limit-note">AI-assisted analysis is limited to 15 changes per selected zone in this prototype. Other detected zones remain visible on the map.</p>
        )}

        <div className="results-filters" role="group" aria-label="Filter results">
          {FILTERS.map((item) => (
            <button
              type="button"
              key={item.id}
              className={filter === item.id ? 'is-active' : ''}
              onClick={() => onFilter(item.id)}
              aria-pressed={filter === item.id}
            >
              {item.label}
              <span>{pad2(counts[item.id])}</span>
            </button>
          ))}
        </div>

        <div className="queue-nav">
          <button type="button" onClick={() => onScroll(-1)} aria-label="Scroll results left" title="Scroll results left">
            <ChevronLeft size={16} strokeWidth={1.8} />
          </button>
          <button type="button" onClick={() => onScroll(1)} aria-label="Scroll results right" title="Scroll results right">
            <ChevronRight size={16} strokeWidth={1.8} />
          </button>
        </div>
      </div>

      <div className="queue-list" ref={queueRef}>
        {filtered.map((candidate) => (
          <CandidateCard
            key={candidate.id}
            candidate={candidate}
            selected={candidate.id === selectedId}
            onSelect={onSelect}
            cardRef={(node) => {
              if (node) cardRefs.current.set(candidate.id, node)
            }}
          />
        ))}
        {!filtered.length && (
          <div className="queue-empty">
            <span>No candidates in this filter.</span>
            <button type="button" onClick={() => onFilter('all')}>Show all results</button>
          </div>
        )}
      </div>
    </section>
  )
}

function EvidenceSheet({
  candidate,
  filteredCount,
  decision,
  onNext,
  onDecide,
  onShowAll,
  sheetScrollRef,
}) {
  if (!candidate) {
    return (
      <aside className="evidence-sheet is-empty" aria-label="No candidates in view">
        <div className="sheet-scroll">
          <span className="sheet-grip" aria-hidden="true" />
          <div className="sheet-empty">
            <h2>No candidates in this filter</h2>
            <p>Nothing in this filter matches the current results. Show all candidates to keep reviewing.</p>
            <button type="button" className="primary-action" onClick={onShowAll}>Show all results</button>
          </div>
        </div>
      </aside>
    )
  }

  const meta = metaFor(candidate)
  return (
    <aside className="evidence-sheet" aria-label={`Evidence for ${candidate.id}`}>
      <div className="sheet-scroll" ref={sheetScrollRef}>
        <span className="sheet-grip" aria-hidden="true" />
        <div className="sheet-head" key={`${candidate.id}-head`}>
          <span className="sheet-id">Candidate {candidate.id.replace('C-', '')} · rank {pad2(candidate.rank)}</span>
          <h2>{candidate.title}</h2>
          <p className="sheet-status">
            {statusLine(candidate)}
            <span> · {meta.label}</span>
          </p>
          <p className="sheet-data">
            D <b>{formatScore(candidate.score)}</b> · {candidate.areaHa.toFixed(1)} ha
            {' · '}{candidate.hasChange ? 'change observed 2017–2024' : candidate.hasChange === false ? 'no visible change' : 'VLM result pending'} · Gemini 2.5 Flash-Lite
          </p>
        </div>

        <div className="sheet-body" key={candidate.id}>
          <CompareViewer candidate={candidate} />
          <div className="model-block">
            <div className="model-block-head">
              <h3>Vision model read</h3>
              <span className="model-conf">{Number.isFinite(candidate.confidence) ? `${candidate.confidence.toFixed(2)} self-reported` : candidate.status}</span>
            </div>
            <p className="model-quote">{candidate.rationale || candidate.error || 'Waiting for the VLM review.'}</p>
          </div>
        </div>
      </div>

      <div className="sheet-actions">
        <button type="button" className="primary-action" onClick={onNext} disabled={filteredCount < 2}>
          Next candidate
          <ArrowRight size={16} strokeWidth={2} />
        </button>
        <button
          type="button"
          className={`decision-button ${decision === 'confirm' ? 'is-confirmed' : ''}`}
          onClick={() => onDecide('confirm')}
          aria-pressed={decision === 'confirm'}
        >
          <Check size={16} strokeWidth={2.1} />
          {candidate.hasChange ? 'Looks right' : 'Agree no change'}
        </button>
        <button
          type="button"
          className={`icon-button icon-button--bordered ${decision === 'flag' ? 'is-flagged' : ''}`}
          onClick={() => onDecide('flag')}
          aria-label="Flag for audit"
          aria-pressed={decision === 'flag'}
          title="Flag for audit"
        >
          <Flag size={16} strokeWidth={1.9} />
        </button>
      </div>

      <div className="sheet-foot">
        <span>Imagery © IGN · Embeddings © Google DeepMind</span>
        <span>D is a screening signal; VLM confidence is self-reported.</span>
      </div>
    </aside>
  )
}

function Toast({ toast }) {
  return (
    <div className="toast" role="status" aria-live="polite" data-visible={Boolean(toast)}>
      {toast && (
        <>
          <span className={`toast-icon toast-icon--${toast.kind}`}>
            {toast.kind === 'flag' ? <Flag size={15} strokeWidth={2} /> : <Check size={15} strokeWidth={2.2} />}
          </span>
          <span className="toast-copy"><b>{toast.title}</b><span>{toast.detail}</span></span>
        </>
      )}
    </div>
  )
}

function App() {
  const [stage, setStage] = useState('draw')
  const [selectedBounds, setSelectedBounds] = useState(ZONE_BOUNDS)
  const [analysis, setAnalysis] = useState(null)
  const [pcaYearsReady, setPcaYearsReady] = useState(false)
  const [comparisonImage, setComparisonImage] = useState(null)
  const [activeLayer, setActiveLayer] = useState('satellite')
  const [comparePosition, setComparePosition] = useState(50)
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [thresholdUpdating, setThresholdUpdating] = useState(false)
  const [selectedId, setSelectedId] = useState(null)
  const [filter, setFilter] = useState('all')
  const [showMethod, setShowMethod] = useState(false)
  const [toast, setToast] = useState(null)
  const [decisions, setDecisions] = useState({})
  const [reviewJob, setReviewJob] = useState(null)
  const [reviewBusy, setReviewBusy] = useState(false)
  const [candidates, setCandidates] = useState([])
  const [scanEpoch, setScanEpoch] = useState(0)
  const cardRefs = useRef(new Map())
  const queueRef = useRef(null)
  const sheetScrollRef = useRef(null)
  const toastTimeout = useRef(null)
  const shellRef = useRef(null)
  const pcaYearsRef = useRef(null)

  useEffect(() => {
    if (!analysis?.alphaearth_2017 || !analysis?.alphaearth_2024) {
      pcaYearsRef.current = null
      setPcaYearsReady(false)
      setComparisonImage(null)
      return undefined
    }
    let cancelled = false
    const loadImage = (base64) => new Promise((resolve, reject) => {
      const image = new Image()
      image.onload = () => resolve(image)
      image.onerror = reject
      image.src = `data:image/png;base64,${base64}`
    })
    Promise.all([loadImage(analysis.alphaearth_2017), loadImage(analysis.alphaearth_2024)])
      .then(([year2017, year2024]) => {
        if (cancelled) return
        pcaYearsRef.current = [year2017, year2024]
        setPcaYearsReady(true)
      })
      .catch(() => {
        if (!cancelled) notify('Could not render AlphaEarth comparison', 'Retry the scan to rebuild the year layers.', 'flag')
      })
    return () => { cancelled = true }
  }, [analysis?.alphaearth_2017, analysis?.alphaearth_2024])

  useEffect(() => {
    const years = pcaYearsRef.current
    if (!analysis || !pcaYearsReady || !years) return
    const [year2017, year2024] = years
    const canvas = document.createElement('canvas')
    canvas.width = year2017.naturalWidth
    canvas.height = year2017.naturalHeight
    const context = canvas.getContext('2d')
    context.drawImage(year2024, 0, 0)
    context.save()
    context.beginPath()
    context.rect(0, 0, canvas.width * comparePosition / 100, canvas.height)
    context.clip()
    context.drawImage(year2017, 0, 0)
    context.restore()
    setComparisonImage(canvas.toDataURL('image/png'))
  }, [analysis?.alphaearth_2017, analysis?.alphaearth_2024, comparePosition, pcaYearsReady])

  useEffect(() => {
    if (!analysis || !pcaYearsReady) return undefined
    setActiveLayer('alphaearth')
    const timers = []
    const velocityTimer = window.setTimeout(() => {
      setActiveLayer('velocity')
      const detectionsTimer = window.setTimeout(() => setActiveLayer('detections'), 1000)
      timers.push(detectionsTimer)
    }, 1000)
    timers.push(velocityTimer)
    return () => timers.forEach((timer) => window.clearTimeout(timer))
  }, [analysis, pcaYearsReady])

  useLayoutEffect(() => {
    const shell = shellRef.current
    if (!shell) return undefined
    const dock = shell.querySelector('.draw-dock, .process-dock, .results-dock')
    if (!dock) {
      shell.style.setProperty('--dock-clearance', '0px')
      return undefined
    }
    const update = () => {
      const rect = dock.getBoundingClientRect()
      const shellRect = shell.getBoundingClientRect()
      const clearance = Math.max(0, shellRect.bottom - rect.top)
      shell.style.setProperty('--dock-clearance', `${clearance}px`)
      shell.style.setProperty('--dock-left-edge', `${Math.max(0, rect.left - shellRect.left)}px`)
      shell.style.setProperty('--dock-right-edge', `${Math.max(0, shellRect.right - rect.right)}px`)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(dock)
    return () => {
      observer.disconnect()
      shell.style.removeProperty('--dock-clearance')
      shell.style.removeProperty('--dock-left-edge')
      shell.style.removeProperty('--dock-right-edge')
    }
  }, [stage])

  const filtered = useMemo(() => {
    if (filter === 'all') return candidates
    if (filter === 'change') return candidates.filter((candidate) => candidate.hasChange)
    return candidates.filter((candidate) => candidate.hasChange === false)
  }, [candidates, filter])

  const counts = {
    all: candidates.length,
    change: candidates.filter((candidate) => candidate.hasChange).length,
    no_change: candidates.filter((candidate) => candidate.hasChange === false).length,
  }

  const selected = filtered.find((candidate) => candidate.id === selectedId) || filtered[0] || null
  const selectedDecision = selected ? decisions[selected.id] : null
  const completedCount = reviewJob?.completed || 0
  const readingIds = reviewJob?.active_id ? [reviewJob.active_id] : []

  useEffect(() => {
    if (!reviewJob?.id || !['queued', 'running'].includes(reviewJob.status)) return undefined
    let cancelled = false
    let timer
    const refresh = async () => {
      try {
        const response = await fetch(`/api/jobs/${reviewJob.id}`)
        const result = await response.json()
        if (!response.ok) throw new Error(result.error || `Queue update failed (${response.status})`)
        if (cancelled) return
        setReviewJob(result)
        setCandidates(result.candidates.map(candidateForUi))
        if (result.status === 'complete' || result.status === 'cancelled') {
          setStage('results')
          setActiveLayer('satellite')
        } else {
          timer = window.setTimeout(refresh, 900)
        }
      } catch (error) {
        if (!cancelled) {
          notify('Queue connection failed', error.message, 'flag')
          timer = window.setTimeout(refresh, 1800)
        }
      }
    }
    timer = window.setTimeout(refresh, 650)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [reviewJob?.id, reviewJob?.status])

  useEffect(() => {
    if (!showMethod) return undefined
    const onKeyDown = (event) => {
      if (event.key === 'Escape') setShowMethod(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [showMethod])

  useEffect(() => {
    if (!filtered.length) return
    if (!filtered.some((candidate) => candidate.id === selectedId)) {
      setSelectedId(filtered[0].id)
    }
  }, [filtered, selectedId])

  useEffect(() => {
    if (window.innerWidth > 900) {
      const card = cardRefs.current.get(selectedId)
      if (card) card.scrollIntoView({ behavior: 'smooth', inline: 'nearest', block: 'nearest' })
    }
    if (sheetScrollRef.current) sheetScrollRef.current.scrollTop = 0
  }, [selectedId, stage])

  useEffect(() => {
    if (!toast) return undefined
    window.clearTimeout(toastTimeout.current)
    toastTimeout.current = window.setTimeout(() => setToast(null), 4200)
    return () => window.clearTimeout(toastTimeout.current)
  }, [toast])

  useEffect(() => () => {
    window.clearTimeout(toastTimeout.current)
  }, [])

  const notify = (title, detail, kind = 'done') => setToast({ title, detail, kind })

  useEffect(() => {
    if (!analysis || threshold === analysis.threshold) {
      setThresholdUpdating(false)
      return undefined
    }
    let cancelled = false
    const timer = window.setTimeout(async () => {
      setThresholdUpdating(true)
      try {
        const response = await fetch('/api/detection-threshold', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ threshold, bounds: selectedBounds }),
        })
        const result = await response.json()
        if (!response.ok) throw new Error(result.error || `Threshold update failed (${response.status})`)
        if (cancelled) return
        setAnalysis((previous) => previous ? { ...previous, ...result } : previous)
        setThresholdUpdating(false)
      } catch (error) {
        if (cancelled) return
        setThreshold(analysis.threshold)
        setThresholdUpdating(false)
        notify('Could not update detection threshold', error.message, 'flag')
      }
    }, 140)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [analysis, threshold, selectedBounds])

  const startScan = async (bounds = ZONE_BOUNDS) => {
    setSelectedBounds(bounds)
    setShowMethod(false)
    setFilter('all')
    setSelectedId(null)
    setDecisions({})
    setReviewJob(null)
    setCandidates([])
    setReviewBusy(false)
    setStage('loading')
    setScanEpoch((value) => value + 1)
    setToast(null)
    setAnalysis(null)
    setComparisonImage(null)
    setActiveLayer('satellite')
    setComparePosition(50)
    setThreshold(DEFAULT_THRESHOLD)
    setThresholdUpdating(false)
    try {
      const response = await fetch('/api/alphaearth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bounds }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || `Download failed (${response.status})`)
      const missingLayer = ['alphaearth_2017', 'alphaearth_2024', 'velocity', 'detections']
        .some((key) => typeof result[key] !== 'string' || !result[key].length)
      const missingDetectionStats = !Number.isFinite(result.threshold)
        || !Number.isFinite(result.detected_pixels)
        || !Number.isFinite(result.detected_components)
        || !Number.isFinite(result.detected_area_ha)
      if (missingLayer || missingDetectionStats) {
        throw new Error('The local API is out of date. Restart app/backend/server.py and retry the scan.')
      }
      setAnalysis(result)
      setStage('detected')
    } catch (error) {
      setStage('draw')
      setActiveLayer('satellite')
      notify('AlphaEarth download failed', `${error.message}. Start the local API and retry.`, 'flag')
    }
  }

  const computeVlmAnalysis = async () => {
    if (!analysis || reviewBusy) return
    setReviewBusy(true)
    try {
      const response = await fetch('/api/vlm-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold, bounds: selectedBounds }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || `Could not start VLM queue (${response.status})`)
      setReviewJob(result)
      setCandidates(result.candidates.map(candidateForUi))
      setSelectedId(null)
      setStage(result.status === 'complete' ? 'results' : 'interpreting')
      setActiveLayer('satellite')
      if (result.total === 0) notify('No regions to review', 'No connected area exceeded 50 pixels at this threshold.', 'flag')
    } catch (error) {
      notify('Could not start VLM analysis', error.message, 'flag')
    } finally {
      setReviewBusy(false)
    }
  }

  const resetToDraw = () => {
    setShowMethod(false)
    setFilter('all')
    setSelectedId(null)
    setDecisions({})
    setReviewJob(null)
    setCandidates([])
    setReviewBusy(false)
    setStage('draw')
    setActiveLayer('satellite')
    setScanEpoch((value) => value + 1)
    setToast(null)
  }

  const selectCandidate = (id) => {
    setSelectedId(id)
    setShowMethod(false)
  }

  const reviewNext = () => {
    if (!filtered.length) return
    const index = filtered.findIndex((candidate) => candidate.id === selectedId)
    const next = filtered[(index + 1) % filtered.length]
    if (next) setSelectedId(next.id)
  }

  const decide = (decision) => {
    if (!selected) return
    setDecisions((previous) => ({ ...previous, [selected.id]: decision }))
    notify(
      decision === 'confirm' ? `${selected.id} confirmed` : `${selected.id} sent to audit`,
      decision === 'confirm' ? 'Your review has been recorded.' : 'The model answer will be checked.',
      decision === 'confirm' ? 'done' : 'flag',
    )
  }

  const scrollQueue = (direction) => {
    if (!queueRef.current) return
    queueRef.current.scrollBy({ left: direction * 380, behavior: 'smooth' })
  }

  const mapCandidates = useMemo(() => {
    if (['interpreting', 'results'].includes(stage)) return candidates.map((candidate) => ({ ...candidate, mapStatus: candidate.mapStatus }))
    return []
  }, [stage, candidates])

  return (
    <div className={`app-shell ${stage === 'interpreting' && selected && ['done', 'error'].includes(selected.status) ? 'has-evidence' : ''}`} data-stage={stage} ref={shellRef}>
      <h1 className="sr-only">Embed2Change land-change review</h1>
      <MapStage
        stage={stage}
        candidates={mapCandidates}
        selectedId={selected ? selected.id : null}
        onSelect={selectCandidate}
        onDrawComplete={startScan}
        scanEpoch={scanEpoch}
        analysis={analysis}
        selectedBounds={selectedBounds}
        comparisonImage={comparisonImage}
        activeLayer={activeLayer}
        onLayerSelect={setActiveLayer}
        comparePosition={comparePosition}
        onCompareChange={setComparePosition}
        threshold={threshold}
        onThresholdChange={setThreshold}
        thresholdUpdating={thresholdUpdating}
      />

      <header className="topbar">
        <button type="button" className="brand" onClick={resetToDraw} aria-label="Embed2Change, start a new scan">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 28 28" fill="none">
              <path d="M4 8.2 14 2l10 6.2v11.6L14 26 4 19.8V8.2Z" stroke="currentColor" strokeWidth="1.1" />
              <path d="m9 11.2 5-3.1 5 3.1v5.6l-5 3.1-5-3.1v-5.6Z" fill="currentColor" />
              <path d="M14 2v6.1M4 8.2l5 3M24 8.2l-5 3M14 19.9V26" stroke="currentColor" strokeWidth="1.1" />
            </svg>
          </span>
          <span className="brand-copy">
            <strong>Embed2Change</strong>
            <span>Prototype</span>
          </span>
        </button>

        <span className="topbar-rule" aria-hidden="true" />

        <div className="job-context">
          <MapPin size={15} strokeWidth={1.8} />
          <span className="job-place">Selected area</span>
          <span className="job-years">2017 → 2024</span>
        </div>

        <div className="top-actions">
          <button
            type="button"
            className={`help-button ${showMethod ? 'is-active' : ''}`}
            onClick={() => setShowMethod((value) => !value)}
            aria-expanded={showMethod}
          >
            <CircleHelp size={17} strokeWidth={1.8} />
            How it works
          </button>
          <button
            type="button"
            className={`replay-button ${stage === 'draw' ? 'is-quiet' : ''}`}
            onClick={resetToDraw}
          >
            <RotateCcw size={15} strokeWidth={1.9} />
            New scan
          </button>
        </div>
      </header>

      {stage === 'draw' && <DrawDock />}

      {['loading', 'classifying', 'detected', 'interpreting'].includes(stage) && (
        <ProcessDock
          key={stage}
          stage={stage}
          completed={completedCount}
          total={candidates.length}
          readingIds={readingIds}
          candidates={candidates}
          onCompute={computeVlmAnalysis}
          busy={reviewBusy}
          analysis={analysis}
          reviewJob={reviewJob}
        />
      )}

      {stage === 'interpreting' && selected && ['done', 'error'].includes(selected.status) && (
        <EvidenceSheet
          candidate={selected}
          filteredCount={filtered.length}
          decision={selectedDecision}
          onNext={reviewNext}
          onDecide={decide}
          onShowAll={() => setFilter('all')}
          sheetScrollRef={sheetScrollRef}
        />
      )}

      {stage === 'results' && (
        <>
          <ResultsDock
            candidates={candidates}
            filtered={filtered}
            selectedId={selected ? selected.id : null}
            filter={filter}
            counts={counts}
            onFilter={setFilter}
            onSelect={selectCandidate}
            onScroll={scrollQueue}
            queueRef={queueRef}
            cardRefs={cardRefs}
            reviewJob={reviewJob}
          />
          <EvidenceSheet
            candidate={filtered.length ? selected : null}
            filteredCount={filtered.length}
            decision={selectedDecision}
            onNext={reviewNext}
            onDecide={decide}
            onShowAll={() => setFilter('all')}
            sheetScrollRef={sheetScrollRef}
          />
        </>
      )}

      {showMethod && <MethodPanel onClose={() => setShowMethod(false)} />}

      <Toast toast={toast} />
    </div>
  )
}

export default App
