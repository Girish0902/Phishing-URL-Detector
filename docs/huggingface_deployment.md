# 🚀 Hugging Face Spaces Deployment Guide

Complete step-by-step guide to deploy your Phishing URL Detection system on Hugging Face Spaces.

## 📋 Prerequisites

- Hugging Face account (free)
- Trained model file (`phishing_url_detector.pkl`)
- All project files ready

## 🎯 Step 1: Create Hugging Face Space

1. **Go to Hugging Face Spaces**
   - Visit: https://huggingface.co/spaces
   - Click "Create new Space"

2. **Configure Space**
   - **Space name:** `phishing-url-detector` (or your choice)
   - **License:** MIT
   - **SDK:** Streamlit
   - **Visibility:** Public
   - **Hardware:** CPU (free tier)

3. **Create Space**
   - Click "Create Space"
   - Wait for initialization (1-2 minutes)

## 📁 Step 2: Upload Project Files

### Method 1: Git Upload (Recommended)

1. **Clone your Space repository**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/phishing-url-detector
   cd phishing-url-detector
   ```

2. **Copy project files**
   ```bash
   # Copy all necessary files
   cp /path/to/your/project/* .
   
   # Required files:
   # - app.py
   # - feature_extractor.py
   # - phishing_url_detector.pkl
   # - requirements.txt
   # - README.md (optional)
   ```

3. **Create app.py (if not already done)**
   ```python
   # Your Streamlit app code here
   import streamlit as st
   # ... rest of your app code
   ```

4. **Update requirements.txt**
   ```txt
   streamlit==1.28.1
   pandas==2.1.4
   numpy==1.26.2
   scikit-learn==1.3.2
   joblib==1.3.2
   requests==2.31.0
   tldextract==5.1.1
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Initial commit: Phishing URL Detector"
   git push origin main
   ```

### Method 2: Web Upload

1. **Go to your Space**
2. **Click "Files" tab**
3. **Click "Add file" → "Upload files"**
4. **Upload files one by one:**
   - `app.py`
   - `feature_extractor.py`
   - `phishing_url_detector.pkl`
   - `requirements.txt`

## ⚙️ Step 3: Configure Space Settings

1. **Go to Space Settings**
   - Click gear icon in top-right
   - Or go to "Settings" tab

2. **Update App Configuration**
   - **Title:** AI Phishing URL Detector
   - **Description:** ML-powered phishing URL detection with explainable results
   - **Color theme:** Blue (or your preference)
   - **Python version:** 3.8+ (default)

3. **Environment Variables** (if needed)
   - No special env vars required for basic setup

## 🔧 Step 4: Test Deployment

1. **Check build logs**
   - Go to "Builds" tab
   - Monitor for errors
   - Common issues:
     - Missing dependencies in requirements.txt
     - Model file not found
     - Import errors

2. **Test the app**
   - Once built, click "App" tab
   - Test with sample URLs:
     - `https://www.google.com` (should be safe)
     - `https://fake-bank-login.com` (should be flagged)

3. **Debug common issues**
   ```bash
   # If build fails, check logs and fix:
   # 1. Ensure all imports are correct
   # 2. Check file paths
   # 3. Verify model file exists
   ```

## 🎨 Step 5: Customize Appearance

### Update README.md for Space

Create or update `README.md`:

```markdown
---
title: AI Phishing URL Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
---

# 🛡️ AI-Based Phishing URL Detector

Detect phishing and malicious URLs using machine learning with explainable risk scoring.

## Features

- Real-time URL analysis
- Risk scoring (0-100)
- Explainable predictions
- 95%+ accuracy
- Clean web interface

## How to Use

1. Enter a URL in the text box
2. Click "Analyze URL"
3. View risk score and analysis

## About

Built with Random Forest classifier trained on 50k+ URLs.
Features extracted: URL structure, domain analysis, suspicious keywords.
```

### Add Space Metadata

The YAML frontmatter in README.md controls:
- Title and emoji
- Color scheme
- SDK and version
- App file location

## 📊 Step 6: Monitor and Maintain

### View Analytics

1. **Go to Space Analytics**
   - Click "Analytics" tab
   - View usage statistics
   - Monitor performance

### Update Model

1. **Retrain model locally**
   ```bash
   python train_model.py
   ```

2. **Upload new model**
   ```bash
   git add phishing_url_detector.pkl
   git commit -m "Update model with new training data"
   git push origin main
   ```

3. **Space auto-rebuilds** with new model

### Handle Issues

**Common Problems & Solutions:**

1. **App not loading**
   - Check build logs for errors
   - Verify all dependencies in requirements.txt
   - Ensure model file is uploaded

2. **Model prediction errors**
   - Check if model file is corrupted
   - Verify feature extraction matches training
   - Test locally first

3. **Slow performance**
   - Optimize feature extraction
   - Consider model compression
   - Upgrade to paid hardware tier

## 🔒 Step 7: Security Considerations

1. **Input Validation**
   - URLs are sanitized in the app
   - No sensitive data stored

2. **Rate Limiting**
   - Hugging Face has built-in rate limiting
   - Consider adding custom limits for heavy usage

3. **Privacy**
   - No user data collected
   - URLs processed in memory only

## 🌟 Step 8: Share and Promote

1. **Get Space URL**
   - Your space URL: `https://huggingface.co/spaces/YOUR_USERNAME/phishing-url-detector`

2. **Share on Social Media**
   - Twitter: "Built an AI phishing detector! Try it: [URL]"
   - LinkedIn: Post about the project with demo

3. **Add to Portfolio**
   - Include in GitHub README
   - Add to resume/projects section

## 💰 Step 9: Upgrade (Optional)

### Paid Features

1. **Hardware Upgrade**
   - CPU → T4 GPU (for faster inference)
   - Cost: $0.60/hour

2. **Custom Domain**
   - Use your own domain
   - Cost: $9/month

3. **Private Spaces**
   - Hide from public
   - Cost: $3/month

### Advanced Features

1. **Persistent Storage**
   - Save user feedback
   - Store analysis history

2. **API Integration**
   - Add REST API endpoints
   - Integrate with other tools

## 📈 Step 10: Scale and Improve

### Performance Optimization

```python
# Add caching for repeated URLs
@st.cache_data
def cached_predict(url):
    return predict_url_risk(url, model)
```

### Add Features

1. **Batch Analysis**
   - Upload CSV of URLs
   - Analyze multiple at once

2. **Export Results**
   - Download analysis reports
   - JSON/CSV export

3. **User Feedback**
   - Allow users to report false positives
   - Improve model over time

## 🐛 Troubleshooting

### Build Errors

**Error: "Module not found"**
```
Solution: Add missing package to requirements.txt
```

**Error: "Model file not found"**
```
Solution: Ensure phishing_url_detector.pkl is uploaded
```

**Error: "Streamlit app not starting"**
```
Solution: Check app.py for syntax errors
```

### Runtime Errors

**Error: "Feature mismatch"**
```
Solution: Ensure feature extraction matches training features
```

**Error: "Memory limit exceeded"**
```
Solution: Optimize code or upgrade hardware
```

## 📞 Support

- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
- **Streamlit Docs:** https://docs.streamlit.io/
- **Community:** Hugging Face Discord or Forums

## ✅ Success Checklist

- [ ] Space created successfully
- [ ] All files uploaded
- [ ] App builds without errors
- [ ] Model predictions work
- [ ] UI is responsive
- [ ] README is informative
- [ ] Space URL shared

## 🎉 You're Done!

Your AI Phishing URL Detector is now live on Hugging Face Spaces! Share the link and showcase your ML project to the world.

**Example Space:** https://huggingface.co/spaces/YOUR_USERNAME/phishing-url-detector
