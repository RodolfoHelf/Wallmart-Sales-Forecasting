# 🚀 GitHub Pages Deployment Guide

This guide will walk you through deploying your Walmart Sales Forecasting Dashboard to GitHub Pages, making it accessible to anyone on the internet.

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Repository with your project code
- ✅ Python 3.9+ installed locally
- ✅ Git configured on your machine

## 🔧 Step-by-Step Deployment

### 1. Prepare Your Repository

First, ensure your repository is properly set up:

```bash
# Clone your repository (if not already done)
git clone https://github.com/RodolfoHelf/Wallmart-Sales-Forecasting.git
cd Wallmart-Sales-Forecasting

# Check current status
git status
```

### 2. Build the Static Site

The project includes a script that automatically converts your FastAPI application to static HTML:

```bash
# Install dependencies (if not already done)
pip install -r config/requirements.txt

# Build the static site
python scripts/build_static_site.py
```

This will:
- ✅ Extract HTML content from `app/main.py`
- ✅ Copy static files (CSS, JS, images)
- ✅ Create `docs/` directory with static site
- ✅ Generate necessary configuration files

### 3. Commit and Push Changes

```bash
# Add all changes
git add .

# Commit the static site
git commit -m "Build static site for GitHub Pages deployment"

# Push to GitHub
git push origin main
```

### 4. Enable GitHub Pages

1. **Go to your repository on GitHub**
2. **Click Settings tab**
3. **Scroll down to Pages section**
4. **Under Source, select "GitHub Actions"**
5. **The workflow will automatically deploy your site**

### 5. Automatic Deployment

The updated GitHub Actions workflow will automatically:
- ✅ Build your static site
- ✅ Deploy directly to GitHub Pages
- ✅ Make your site live

**Your dashboard will be available at:**
```
https://rodolfoh.github.io/Wallmart-Sales-Forecasting
```

## 🔄 Updating Your Site

### Making Changes

1. **Edit your dashboard content** in `app/main.py`
2. **Update styles** in `app/static/css/style.css`
3. **Add functionality** in `app/static/js/main.js`
4. **Test locally** by running the FastAPI app

### Deploying Updates

```bash
# Build the updated static site
python scripts/build_static_site.py

# Commit and push changes
git add .
git commit -m "Update dashboard content and styling"
git push origin main
```

The GitHub Actions workflow will automatically rebuild and deploy your updated site.

## 🛠️ Customization

### Changing Site Configuration

Edit `docs/_config.yml` to customize:
- Site title and description
- Social media links
- Repository-specific settings

### Adding New Pages

1. **Create new HTML files** in the `docs/` directory
2. **Update navigation** in your main dashboard
3. **Rebuild the static site**
4. **Push changes**

### Custom Domain (Optional)

1. **Purchase a domain** (e.g., `yourdomain.com`)
2. **Add CNAME record** pointing to `rodolfoh.github.io`
3. **Create `docs/CNAME` file** with your domain
4. **Wait for DNS propagation** (up to 24 hours)

## 🔍 Troubleshooting

### Common Issues

#### Site Not Loading
- ✅ Check if GitHub Pages is enabled
- ✅ Verify GitHub Actions workflow completed successfully
- ✅ Wait a few minutes for deployment

#### Styling Issues
- ✅ Ensure static files are copied correctly
- ✅ Check browser console for errors
- ✅ Verify CSS paths are relative (`./static/`)

#### Build Failures
- ✅ Check GitHub Actions logs
- ✅ Verify Python dependencies are installed
- ✅ Ensure all required files exist

### Debugging Steps

1. **Check GitHub Actions**:
   - Go to Actions tab in your repository
   - Review the latest workflow run
   - Check for error messages

2. **Verify File Structure**:
   ```bash
   # Check docs directory contents
   ls -la docs/
   
   # Verify static files
   ls -la docs/static/
   ```

3. **Test Locally**:
   ```bash
   # Serve static files locally
   cd docs
   python -m http.server 8000
   # Open http://localhost:8000
   ```

## 📱 Mobile Optimization

Your dashboard is already optimized for mobile devices with:
- ✅ Responsive design
- ✅ Touch-friendly navigation
- ✅ Mobile-optimized layouts
- ✅ Fast loading times

## 🚀 Performance Tips

### Optimize Images
- Use WebP format when possible
- Compress images before adding
- Consider lazy loading for large images

### Minimize CSS/JS
- Remove unused CSS rules
- Minify JavaScript files
- Use CDN for external libraries

### Caching
- GitHub Pages automatically handles caching
- Static assets are cached aggressively
- Consider versioning for cache busting

## 🔒 Security Considerations

- ✅ No sensitive data in static files
- ✅ Use environment variables for API keys
- ✅ Validate all user inputs
- ✅ Keep dependencies updated

## 📊 Analytics (Optional)

### Google Analytics
1. **Create Google Analytics account**
2. **Get tracking code**
3. **Add to `docs/_config.yml`**
4. **Rebuild and deploy**

### GitHub Insights
- View traffic statistics in repository Insights
- Monitor popular pages and referrers
- Track visitor engagement

## 🎯 Next Steps

After successful deployment:

1. **Share your dashboard** with stakeholders
2. **Collect feedback** and iterate
3. **Add more interactive features**
4. **Integrate with real-time data sources**
5. **Expand to multiple pages**

## 📞 Support

If you encounter issues:

- **GitHub Issues**: Create an issue in your repository
- **GitHub Discussions**: Start a discussion for help
- **Documentation**: Check this guide and project README
- **Community**: Reach out to the open-source community

---

🎉 **Congratulations! Your Walmart Sales Forecasting Dashboard is now live on GitHub Pages!**

Share your success: `https://rodolfoh.github.io/Wallmart-Sales-Forecasting`
