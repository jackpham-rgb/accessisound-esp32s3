# How to Upload This to GitHub (Step by Step)

## Step 1 — Create a new repo on GitHub
1. Go to https://github.com/new
2. Repository name: `accessisound-esp32s3`
3. Description: `ESP32-S3 accessibility sound recognition assistant — TFLite Micro on-device inference with haptic/audio alerts`
4. Set to **Public**
5. ✅ Check "Add a README file" → **NO** (we have our own)
6. Click **Create repository**

## Step 2 — Push from your computer
Open Terminal / Command Prompt in the folder you extracted:

```bash
cd accessisound-esp32s3

# Initialise git
git init
git add .
git commit -m "feat: initial hackathon prototype — ESP32-S3 sound classifier"

# Link to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/accessisound-esp32s3.git
git branch -M main
git push -u origin main
```

## Step 3 — Add a GitHub topic tags
After pushing, go to your repo page → gear icon next to "About":
- Add topics: `esp32`, `embedded-ml`, `tflite-micro`, `accessibility`, `sound-classification`, `arduino`, `hackathon`

## Step 4 — Enable GitHub Pages for the README (optional)
Settings → Pages → Source: main branch → /docs → Save

## Step 5 — Add to your portfolio website
Copy the `portfolio-project-card.html` snippet and embed it in your site.
Update the GitHub link inside the card to your actual repo URL.
