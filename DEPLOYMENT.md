# Deploying Movie Recommendation App to Render

This guide provides step-by-step instructions for deploying the Movie Recommendation application to Render.

## Prerequisites

1. A GitHub account
2. A Render account (sign up at [render.com](https://render.com))
3. TMDB API key (can be obtained from [themoviedb.org](https://www.themoviedb.org/settings/api))

## Deployment Options

### Option 1: One-Click Deployment (Recommended)

1. Fork this repository to your GitHub account
2. In your forked repository, update the `render.yaml` file with your GitHub repository URL
3. In Render dashboard, go to "Blueprints" and click on "New Blueprint Instance"
4. Connect your GitHub account and select your forked repository
5. Configure environment variables:
   - `TMDB_API_KEY`: Your TMDB API key
6. Review the settings and click "Apply"
7. Render will automatically deploy your application

### Option 2: Manual Deployment

1. Push your code to GitHub
2. Make the build script executable:
   ```bash
   chmod +x render-build.sh
   git add render-build.sh
   git update-index --chmod=+x render-build.sh
   git commit -m "Make render-build.sh executable"
   git push
   ```
3. In Render dashboard, go to "Web Services" and click "New Web Service"
4. Connect your GitHub account and select your repository
5. Configure the service:
   - Environment: Docker
   - Region: Choose based on your location
   - Branch: main (or your preferred branch)
   - Plan: Free (or choose a paid plan)
   - Environment Variables:
     - `TMDB_API_KEY`: Your TMDB API key
     - `ENV`: production
6. Click "Create Web Service"

### Option 3: CI/CD with GitHub Actions

For automatic deployments using GitHub Actions:

1. Set up two GitHub secrets in your repository settings:
   - `RENDER_API_KEY`: Your Render API key (from Render dashboard)
   - `RENDER_SERVICE_ID`: Your Render service ID (from Render dashboard)
2. Push to your main branch to trigger automatic deployment

## Verifying the Deployment

1. Once deployment is complete, Render will provide a URL for your application
2. Open the URL in your browser to access your application
3. Check the `/health` endpoint to verify the API is running properly
4. Test the movie recommendation functionality

## Troubleshooting

If you encounter any issues during deployment:

1. Check the build logs in Render dashboard
2. Verify the `TMDB_API_KEY` environment variable is set correctly
3. Make sure the data download was successful
4. Check if the Docker build completed successfully

## Data Persistence

The application uses the data file downloaded during the build process. If you need to update the dataset:

1. Update the URL in `render-build.sh`
2. Commit and push the changes
3. Redeploy the application

## Common Issues

### Missing Data File

If the application fails to start due to missing data file:
- Check the build logs to see if the download was successful
- Try manually downloading the file and adding it to your repository
- Verify the Google Drive link is still valid

### TMDB API Key Issues

If the application cannot fetch movie data:
- Verify your TMDB API key is valid
- Check that the environment variable is properly set in Render
- Look for any rate limiting messages in the application logs

### Frontend Not Loading

If the frontend UI is not loading:
- Check if the application is properly serving static files
- Verify that the build process completed successfully
- Check the browser console for any errors 