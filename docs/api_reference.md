# SmartBite API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /health

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "model": "smartbite-v1",
  "backend": "pytorch"
}
```

### POST /analyze

Analyze a single food image.

**Request:** multipart/form-data with `file` field (image/jpeg, image/png)

**Response:**
```json
{
  "food_class": "142",
  "freshness": {
    "overall": 0.872,
    "appearance": 0.831,
    "texture": 0.764,
    "color": 0.912,
    "spoilage_level": "NONE",
    "quality_grade": "A",
    "confidence": 0.943
  },
  "inference_time_ms": 45.2
}
```

### POST /analyze/batch

Analyze multiple food images (up to 100 per request).

**Request:** multipart/form-data with multiple `files` fields

**Response:**
```json
{
  "results": [...],
  "count": 10
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Invalid image or missing file |
| 413 | Image too large (>16MB) |
| 500 | Internal inference error |

## Rate Limits

- Free tier: 100 req/day
- Pro tier: 10,000 req/day
- Enterprise: Custom SLA
