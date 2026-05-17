# EduAssess Python OMR Microservice

This repository now exposes your OMR scanner as an HTTP microservice so your Laravel app (Hostinger) can call it.

## 1. API Endpoints

- `GET /health`
- `POST /scan/exam` (single pipeline: perspective warp, normalize, detect bubbles)
- `POST /scan/term` (same pipeline for term sheets)
- `POST /scan/bubbles` (bubble-only compatibility endpoint)
- Laravel-compatible aliases:
  - `POST /api/entrance/omr/check`
  - `POST /api/instructor/omr/check-term`
  - `POST /api/entrance/omr/check-bubbles`

For scan endpoints:
- Body: `multipart/form-data`
- Field name: `file`
- Header: `X-API-Key: <your-key>` (required only if `OMR_API_KEY` is set)

## 2. Local Run

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

## 3. Deploy To Railway (Step by Step)

1. Push this repo to GitHub.
2. In Railway dashboard, create a new project.
3. Choose **Deploy from GitHub repo** and select this repository.
4. Open your service settings and set the start command to:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

5. Add environment variables:
   - `OMR_API_KEY` = strong secret key
   - `OMR_DEBUG_DIR` = `/tmp/debug`
   - Optional: `OMR_MAX_DIM` and `OMR_REDUCED_FACTOR`
6. Go to Networking and generate a public domain.
7. Open `https://<your-railway-domain>/health` and check it returns `{"ok":true}`.

## 4. Connect Laravel (Hostinger) To Railway

### `.env` in Laravel

```env
OMR_SERVICE_URL=https://<your-railway-domain>
OMR_SERVICE_KEY=<same-value-as-OMR_API_KEY>
```

### `config/services.php`

```php
'omr' => [
    'url' => env('OMR_SERVICE_URL'),
    'key' => env('OMR_SERVICE_KEY'),
],
```

### Example Laravel call

```php
use Illuminate\Support\Facades\Http;

$response = Http::timeout(90)
    ->retry(2, 500)
    ->withHeaders([
        'X-API-Key' => config('services.omr.key'),
    ])
    ->attach('file', fopen($absoluteImagePath, 'r'), basename($absoluteImagePath))
    ->post(config('services.omr.url') . '/scan/exam');

$result = $response->throw()->json();
```

### Response shape

```json
{
  "sheet_id": "QR value or null",
  "answers": {"1": "A", "2": "invalid"},
  "debug": "omr_processed/<file>_04_after_check_bubbles.jpg",
  "debug_images": {
    "perspective_warp": "omr_processed/<file>_02_perspective_warp.jpg",
    "after_check_bubbles": "omr_processed/<file>_04_after_check_bubbles.jpg"
  }
}
```

## 5. Notes

- Railway filesystem is ephemeral, so debug images in `/tmp/debug` are temporary.
- If your scans are large/high-volume, keep your service on a Railway plan with enough RAM/CPU.
- Keep `OMR_API_KEY` private and rotate it if leaked.
