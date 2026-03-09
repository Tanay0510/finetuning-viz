# Deploy to Vercel (FREE) with Squarespace Domain

Total cost: $0. Vercel free tier includes custom domains + HTTPS.

---

## Step 1: Push to GitHub

```bash
cd finetuning-viz
git init
git add .
git commit -m "LLM fine-tuning 3D visualizer"

# Create a repo at github.com/new called "finetuning-viz"
git remote add origin https://github.com/YOUR_USERNAME/finetuning-viz.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Vercel

1. Go to **https://vercel.com** → Sign up with GitHub (free)
2. Click **"Add New Project"**
3. Import your `finetuning-viz` repo
4. Framework preset: **Other** (Vercel auto-detects via `vercel.json`)
5. Click **Deploy**
6. Wait ~1 minute → Vercel gives you a URL like:
   `https://finetuning-viz.vercel.app`
7. Visit it — your site should be live

## Step 3: Add Your Squarespace Domain in Vercel

1. In Vercel dashboard → your project → **Settings** → **Domains**
2. Type your domain (e.g. `yourdomain.com`) → click **Add**
3. Vercel shows you the DNS records you need. Typically:
   - **A Record**: `76.76.21.21`
   - **CNAME for www**: `cname.vercel-dns.com`

## Step 4: Configure Squarespace DNS

1. Log into **Squarespace** → **Domains** → click your domain
2. Click **DNS** → **DNS Settings**
3. **Delete the Squarespace default records** (trash icon on each one)
   - Keep MX records if you have email on this domain
4. Scroll to **Custom Records** → **Add Record**

**Record 1 — Root domain:**
```
Type: A
Host: @ (leave blank if @ doesn't work)
Data: 76.76.21.21
```

**Record 2 — www subdomain:**
```
Type: CNAME
Host: www
Data: cname.vercel-dns.com
```

5. Click **Save**

## Step 5: Wait and Verify

- DNS takes 5 minutes to 48 hours to propagate (usually around 15 min)
- Vercel auto-provisions HTTPS (SSL) — no extra steps
- Check propagation: https://dnschecker.org

Visit `https://yourdomain.com` — your 3D visualizer should be live!

---

## Updating Your Site

After any code changes:

```bash
git add .
git commit -m "description of changes"
git push
```

Vercel auto-deploys in around 30 seconds. That's it.

---

## Troubleshooting

**404 Not Found after deploy:**
- Make sure `vercel.json` is in the root of your repo
- Make sure `app.py` has `app = Flask(__name__)` (Vercel looks for `app`)

**Domain not working:**
- Verify DNS records match exactly what Vercel shows
- Check https://dnschecker.org for propagation
- In Vercel → Settings → Domains — it should say "Valid Configuration"

**502 Bad Gateway:**
- Check Vercel function logs: Project → Deployments → click latest → Logs
- Common fix: make sure `requirements.txt` lists `flask`
