# Fix Git Push Issue - Windows PowerShell Commands

## Problem
Git is trying to push large model files (>100MB) that exceed GitHub's file size limit.

## Solution for Windows PowerShell

### Step 1: Navigate to the SMS_ham_Spam folder
```powershell
cd SMS_ham_Spam
```

### Step 2: Remove git history (where large files are stuck)
```powershell
# PowerShell command to remove .git folder
Remove-Item -Recurse -Force .git
```

### Step 3: Start fresh git repository
```powershell
git init
```

### Step 4: Add all files (models will be ignored by .gitignore)
```powershell
git add .
```

### Step 5: Check what will be committed (should NOT see models/)
```powershell
git status
```

You should see:
- ✅ Source code files (.py)
- ✅ Data files (.csv)
- ✅ Documentation files (.md, .txt)
- ✅ .gitignore
- ❌ NO models/ folder
- ❌ NO .pkl or .pt files

### Step 6: Commit everything
```powershell
git commit -m "Initial commit: SMS Spam Classification Project"
```

### Step 7: Add your GitHub repository as remote
```powershell
# Replace YOUR_USERNAME and YOUR_REPO with your actual values
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Step 8: Push to GitHub
```powershell
# For new repository or first push
git push -u origin main

# If it says 'main' branch doesn't exist, you might be on 'master'
# Check with:
git branch

# If you're on master:
git branch -M main
git push -u origin main
```

## Complete Copy-Paste Commands

```powershell
# Make sure you're in SMS_ham_Spam folder
cd SMS_ham_Spam

# Remove old git history
Remove-Item -Recurse -Force .git

# Start fresh
git init

# Add files (models excluded via .gitignore)
git add .

# Check what's being added
git status

# Commit
git commit -m "Initial commit: SMS Spam Classification Project

This project implements binary SMS spam classification using:
- Classical ML: TF-IDF + Naive Bayes (baseline)
- Modern ML: DistilBERT (transformer)
- Comprehensive evaluation and comparison
- Full reproducibility (models generated via training scripts)

Note: Model files excluded from repo due to size (>1GB).
Run training scripts to reproduce models."

# Add your remote (REPLACE WITH YOUR REPO URL)
# git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push
# git push -u origin main
```

## Verify Before Pushing

### Check repository size
```powershell
# Check .git folder size (should be small)
(Get-ChildItem .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
```

Should show < 5 MB

### Check what files are tracked
```powershell
git ls-files
```

You should see:
- ✅ .gitignore
- ✅ README.md
- ✅ requirements.txt
- ✅ src/*.py files
- ✅ data/processed/*.csv
- ❌ Nothing in models/

## Alternative: If you don't want to lose git history

If you have important commit history you want to keep, use this instead:

```powershell
# Remove models from tracking but keep files locally
git rm -r --cached models/

# Add .gitignore
git add .gitignore

# Commit the changes
git commit -m "Remove model files from git tracking"

# Use BFG Repo-Cleaner to remove from history
# Download BFG from: https://rtyley.github.io/bfg-repo-cleaner/

# Then run:
# java -jar bfg.jar --delete-folders models --no-blob-protection .
```

But for a new project, the fresh start approach is simpler!

## Troubleshooting

### Error: "fatal: not a git repository"
Good! This means the .git folder was removed successfully. Continue with `git init`.

### Error: "remote origin already exists"
```powershell
git remote remove origin
git remote add origin YOUR_REPO_URL
```

### Error: "failed to push some refs"
```powershell
# For brand new repo, force push is safe
git push -u origin main --force
```

### Still see large files in git status?
Check your .gitignore is working:
```powershell
git check-ignore -v models/
```

Should show: `.gitignore:15:models/    models/`

## Success Indicators

After pushing, your GitHub repository should show:
- Repository size: ~2 MB
- No model files (.pkl, .pt, .safetensors)
- Clear documentation explaining models are reproducible
- All necessary code and data to regenerate models

You're done! 🎉

