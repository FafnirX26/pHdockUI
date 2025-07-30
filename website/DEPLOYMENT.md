# Vercel Deployment Guide

## Issues Fixed

### 1. Incorrect `vercel.json` Configuration
- **Problem**: The `outputDirectory` was set to `website/.next` instead of `.next`
- **Solution**: Updated `vercel.json` to use the correct path

### 2. Missing Environment Variables
- **Problem**: The app uses `NEXT_PUBLIC_API_URL` but it wasn't configured for production
- **Solution**: Added proper environment variable handling in `next.config.ts`

### 3. Next.js Configuration
- **Problem**: Missing proper production configuration
- **Solution**: Added `output: 'standalone'` and proper environment variable handling

## Deployment Steps

### 1. Environment Variables Setup
In your Vercel dashboard, add the following environment variable:
- **Name**: `NEXT_PUBLIC_API_URL`
- **Value**: Your backend API URL (e.g., `https://your-backend-domain.com`)

### 2. Backend Deployment
The frontend depends on a backend API. You need to deploy the backend separately and update the `NEXT_PUBLIC_API_URL` environment variable.

### 3. Vercel Configuration
The `vercel.json` file is now properly configured:
```json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "outputDirectory": ".next",
  "framework": "nextjs"
}
```

### 4. Next.js Configuration
The `next.config.ts` file now includes:
- Standalone output for better deployment
- Proper environment variable handling
- Server external packages configuration

## Testing Deployment

1. **Local Build Test**: Run `npm run build` to ensure the build works locally
2. **Environment Variables**: Make sure `NEXT_PUBLIC_API_URL` is set in Vercel
3. **Backend Availability**: Ensure your backend API is deployed and accessible

## Common Issues

1. **Build Failures**: Usually caused by missing dependencies or TypeScript errors
2. **Runtime Errors**: Often related to missing environment variables
3. **API Connection Issues**: Backend not deployed or incorrect API URL

## Troubleshooting

If deployment still fails:
1. Check Vercel build logs for specific error messages
2. Verify all environment variables are set correctly
3. Ensure the backend API is accessible from Vercel's servers
4. Test the build locally with `npm run build` 