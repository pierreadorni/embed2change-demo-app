# AlphaEarth demo API

The frontend requests the fixed Plaine Saint-Denis demo window from the public
Source Cooperative AlphaEarth GeoZarr. The API reads only the 2017 and 2024
window, dequantizes and normalizes the paired 64-band embeddings, computes one
shared three-component PCA for both years, and calculates total displacement
`D = ||e_2024 - e_2017||₂`. It returns the two PCA views, annualized displacement
(`D / 7`), and the threshold mask (`D >= 0.45`) for map layers. The detection
mask keeps only 8-connected components larger than 50 pixels.

Fetched source pixel windows are cached as separate 2017 and 2024 `.npy` files
under `backend/cache/alphaearth/tiles-v3/`. Each requested chunk is checked
against cached pixel extents; covered portions are read from memory-mapped files
and only uncovered runs are downloaded and added to the cache. This allows
click-and-drag zones and their 5 km chunks to reuse overlapping source pixels
even when their geographic bounds differ. Set `ALPHAEARTH_CACHE_DIR` to use
another location. Remove `tiles-v3` to force fresh downloads. Older v2 entries
are left in place and are not read by this cache format. PCA and displacement
are calculated in small row blocks. The first chunk supplies the shared PCA
mean, components, and colour limits used to render every later chunk in the
same RGB space.

Start the API from the repository root:

```sh
<notebook-python> app/backend/server.py
```

```sh
cd app/frontend
npm run dev
```

Install `app/backend/requirements.txt` into the Python environment if needed.
The Vite development server proxies `/api` to `127.0.0.1:8000`. Scanning needs
internet access to Source Cooperative S3. Hosted Gemini review needs IGN
Géoplateforme and a Google AI Studio API key. Local Gemma inference is disabled
in this version; its implementation remains commented in `review_queue.py` for
possible reactivation later.

After the scan, the map layer selector offers the IGN satellite view, a swipe
comparison of 2017 and 2024 AlphaEarth PCA, annualized displacement, and the
live threshold mask. The detection slider ranges from `0.10` to `1.20`; changes
reuse the in-memory displacement raster and do not re-download embeddings. After
choosing a threshold, select **Compute VLM analysis** to queue connected regions.
The worker downloads 2017 and 2024 IGN crops, outlines each region, concatenates
the pair, and sends it to Google AI Studio, which uses Gemini 2.5 Flash-Lite through the
`google-genai` client; its key is read from `GOOGLE_AI_STUDIO_API_KEY`
(or `GOOGLE_API_KEY` / `GEMINI_API_KEY`)
in the repository-root `.env` file. Gemini 2.5 Flash-Lite supports image input
and is the lowest-cost general vision model in Google's current pricing table.
The worker reviews up to four regions concurrently, downloading imagery and
requesting Gemini analysis in parallel. Queue status updates for each active
region; completed regions are selectable on the map while the worker continues.
Jobs are held in memory and cleared when the API restarts. The demo queues up to
the 15 largest regions, matching the per-zone AI analysis limit.
