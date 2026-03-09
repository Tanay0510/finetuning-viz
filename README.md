# LLM Fine-Tuning 3D Visualizer

Interactive 3D visualization of LLM fine-tuning methods — Full FT, LoRA, QLoRA, Prefix Tuning, Adapters, RLHF/DPO.

Built with Flask + Three.js. Neural network graph rendered in 3D with animated data flow, method-specific architecture, and orbital camera controls.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Project Structure

```
finetuning-viz/
├── app.py              # Flask app + method data (edit methods here)
├── templates/
│   └── index.html      # Full page with Three.js visualization
├── static/             # (add custom CSS/JS/images here)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Editing Method Data

All method data lives in `app.py` in the `METHODS` list. Each method has:

- `id`, `name`, `short`, `emoji` — identity
- `color`, `accent` — theme colors (hex)
- `desc` — what the 3D scene shows
- `stats` — key metrics dict
- `formula` — mathematical formulation
- `insight` — key takeaway text

Edit the Python dict and the page updates automatically.

## Deployment Options

### Option 1: Direct (VPS, EC2, DigitalOcean, etc.)

```bash
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:8000 --workers 2
```

Then put Nginx in front:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Option 2: Docker

```bash
docker build -t finetuning-viz .
docker run -p 8000:8000 finetuning-viz
```

### Option 3: Railway / Render / Fly.io

Just push the repo — these platforms auto-detect the Dockerfile or Procfile.

Create a `Procfile` if needed:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### Option 4: Vercel (serverless)

Create `vercel.json`:
```json
{
  "builds": [{"src": "app.py", "use": "@vercel/python"}],
  "routes": [{"src": "/(.*)", "dest": "app.py"}]
}
```

## Adding HTTPS (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## Tech Stack

- **Backend**: Flask (Python 3.11+)
- **3D Engine**: Three.js r128 (CDN, no build step)
- **Fonts**: Outfit + Fira Code (Google Fonts)
- **Deployment**: Gunicorn + Nginx (or Docker)
