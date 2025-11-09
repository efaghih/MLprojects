# Remove Nested Git Repository

## Goal
Remove the git repository from SMS_ham_Spam and use the parent folder's git repository instead.

## Windows PowerShell Commands

### Step 1: Remove git from SMS_ham_Spam folder
```powershell
# Navigate to SMS_ham_Spam
cd SMS_ham_Spam

# Remove the .git folder
Remove-Item -Recurse -Force .git

# Go back to parent directory
cd ..
```

### Step 2: Update .gitignore in parent folder
```powershell
# You're now in MLprojects folder
# Add rules to ignore large model files
```

Add this to `MLprojects/.gitignore` (or create it if it doesn't exist):
```
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual Environment
ml_env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/

# Model files (too large - regenerate using training scripts)
SMS_ham_Spam/models/
*.pkl
*.pth
*.pt
*.safetensors
*.bin

# Large reports (optional - can regenerate)
SMS_ham_Spam/reports/*.png

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

### Step 3: Add and commit from parent directory
```powershell
# Make sure you're in MLprojects folder
cd MLprojects  # or wherever your parent git repo is

# Check git status
git status

# Add SMS_ham_Spam folder (models will be ignored)
git add SMS_ham_Spam/

# Commit
git commit -m "Add SMS Spam Classification project"

# Push
git push origin main
```

## Complete Copy-Paste Solution

```powershell
# Navigate to SMS_ham_Spam and remove its git
cd SMS_ham_Spam
Remove-Item -Recurse -Force .git
cd ..

# Now you're in the parent directory (MLprojects)
# Check if git exists here
git status

# Add .gitignore rules (create file if needed)
# Use your text editor or:
@"
__pycache__/
*.py[cod]
ml_env/
venv/
SMS_ham_Spam/models/
*.pkl
*.pt
*.safetensors
"@ | Out-File -FilePath .gitignore -Append -Encoding utf8

# Add the SMS_ham_Spam folder
git add SMS_ham_Spam/

# Check what will be committed (should NOT include models/)
git status

# Commit
git commit -m "Add SMS Spam Classification project

- Classical ML baseline (TF-IDF + Naive Bayes)
- Modern transformer baseline (DistilBERT)
- Comprehensive evaluation and comparison
- Model files excluded (>1GB) - regenerate via training scripts"

# Push
git push origin main
```

## Verify Before Pushing

### Check what's being added:
```powershell
git status
```

Should see:
- ✅ SMS_ham_Spam/*.py
- ✅ SMS_ham_Spam/src/*.py
- ✅ SMS_ham_Spam/data/processed/*.csv
- ✅ SMS_ham_Spam/*.md
- ❌ NOT SMS_ham_Spam/models/

### Check repository size:
```powershell
git count-objects -vH
```

Should be reasonable size (< 10 MB)

## If Parent Folder Doesn't Have Git

If the parent folder (MLprojects) doesn't have a git repository yet:

```powershell
# Navigate to parent folder
cd MLprojects

# Initialize git
git init

# Create .gitignore (see content above)
# Then add the ignore rules for models

# Add everything
git add .

# Commit
git commit -m "Initial commit: ML Projects repository"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/MLprojects.git

# Push
git push -u origin main
```

## Advantages of This Approach

✅ **Single repository** for all your ML projects
✅ **Cleaner structure** - MLprojects/SMS_ham_Spam/
✅ **Easier management** - one place to track everything
✅ **No nested git issues**
✅ **Can add more projects** later to the same repo

## Final Structure

```
MLprojects/                    (git repository here)
├── .git/                      (root level)
├── .gitignore                 (ignores models/)
├── SMS_ham_Spam/
│   ├── .gitignore             (can keep for local ignores)
│   ├── src/
│   ├── data/
│   ├── models/                (IGNORED by git)
│   ├── *.py
│   └── *.md
├── linearRg/
├── Scikit-Practice/
└── vectorization/
```

This way, your entire MLprojects folder is one repository! 🎯

