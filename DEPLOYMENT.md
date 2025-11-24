# Deployment Guide: Vercel + Railway

This guide covers deploying pHdockUI with Vercel (frontend) and Railway (backend).

## Prerequisites

- [Vercel CLI](https://vercel.com/docs/cli) installed
- [Railway CLI](https://docs.railway.app/guides/cli) installed and logged in
- Git repository cloned and owned by you

## Railway Backend Deployment

### Step 1: Initialize Railway Project

Since Railway CLI requires interactive input, you'll need to run these commands manually:

```bash
# Initialize Railway project (will prompt for workspace and project name)
railway init

# This will ask you to:
# 1. Select your workspace
# 2. Choose "Create new project"
# 3. Name your project (e.g., "phdockui-backend")
```

### Step 2: Set Environment Variables

```bash
# Set the port (Railway will override with $PORT automatically)
railway variables set PORT=8000

# Optional: Add your Vercel frontend URL for CORS
railway variables set FRONTEND_URL=https://your-app.vercel.app
```

### Step 3: Deploy to Railway

```bash
# Deploy using the Railway Dockerfile
railway up

# Or link to GitHub and enable auto-deploys
railway link
```

### Step 4: Get Your Railway URL

```bash
# Generate a public domain
railway domain

# This will give you a URL like: https://phdockui-backend-production.up.railway.app
```

## Vercel Frontend Deployment

### Step 1: Configure Environment Variables

Before deploying to Vercel, you need to set the backend API URL.

1. Go to your Vercel project settings
2. Navigate to "Environment Variables"
3. Add the following:

```
NEXT_PUBLIC_API_URL=https://your-railway-app.up.railway.app
```

Or via Vercel CLI:

```bash
cd website
vercel env add NEXT_PUBLIC_API_URL
# Enter your Railway backend URL when prompted
```

### Step 2: Deploy to Vercel

```bash
cd website

# Login to Vercel (if not already)
vercel login

# Deploy
vercel --prod
```

### Step 3: Update CORS in Backend

After getting your Vercel URL, you need to update the CORS configuration:

1. Edit `website/backend/main.py`
2. Add your Vercel URL to the `allow_origins` list:

```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:3001",
    "https://ph-dock.vercel.app",
    "https://your-actual-vercel-url.vercel.app"  # Add this
],
```

3. Commit and push changes to trigger Railway redeploy

## Configuration Files Created

- **`railway.toml`**: Railway configuration in project root
- **`railway.dockerfile`**: Optimized Dockerfile for Railway deployment
- **`.env.example`**: Environment variable template

## Architecture

```
[User Browser]
    ↓
[Vercel - Next.js Frontend] (your-app.vercel.app)
    ↓ API calls
[Railway - FastAPI Backend] (your-app.up.railway.app)
    ↓
[ML Models & pKa Prediction]
```

## Important Notes

1. **Models Directory**: Make sure the `models/` directory with trained models is committed to your repository or uploaded to Railway
2. **CORS**: The backend CORS must include your Vercel deployment URL
3. **Environment Variables**: Frontend needs `NEXT_PUBLIC_API_URL` pointing to Railway backend
4. **Port Configuration**: Railway automatically sets `$PORT`, the Dockerfile is configured to use it

## Troubleshooting

### Backend Deploy Fails
- Check Railway logs: `railway logs`
- Verify all dependencies in `requirements.txt` are installable
- Ensure `src/` and `models/` directories are in repository

### Frontend Can't Connect to Backend
- Verify `NEXT_PUBLIC_API_URL` is set in Vercel environment variables
- Check CORS configuration in `website/backend/main.py`
- Ensure Railway backend is running: `railway status`

### CORS Errors
- Add your Vercel URL to `allow_origins` in `website/backend/main.py`
- Redeploy backend after CORS changes

## Quick Commands Reference

```bash
# Railway
railway login              # Login to Railway
railway init               # Initialize project
railway up                 # Deploy
railway logs               # View logs
railway status             # Check status
railway domain             # Generate public domain

# Vercel
vercel login               # Login to Vercel
vercel                     # Deploy to preview
vercel --prod              # Deploy to production
vercel logs                # View logs
vercel env ls              # List environment variables
```

## Next Steps After Deployment

1. Test the deployed application
2. Set up custom domains (optional)
3. Configure monitoring and alerts
4. Set up CI/CD with GitHub integration
5. Consider adding Redis for job queue in Railway
