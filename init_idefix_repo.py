import os
import subprocess

REPO_PATH = "/tiara/home/mlehmann/data/idefix-mkl"
REMOTE_URL = "git@github.com:mlehmann86/IDEFIX-ml.git"
IGNORE_CONTENT = """\
outputs/
reference/
*.out
*.err
*.txt
*.dat
*.vtk
*.log
*.pyc
*.o
*.mod
cmake-build*/
build/
CMakeFiles/
CMakeCache.txt
Makefile
submit*.sh
"""

def run(cmd):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    os.chdir(REPO_PATH)

    # Remove previous .git if messed up
    if os.path.isdir(".git"):
        print("⚠️  .git exists — resetting it.")
        run("rm -rf .git")

    print("🧼 Initializing Git...")
    run("git init")
    run(f"git remote add origin {REMOTE_URL}")
    run("git branch -M main")

    print("🧹 Writing .gitignore...")
    with open(".gitignore", "w") as f:
        f.write(IGNORE_CONTENT)

    print("🪓 Removing reference submodule if present...")
    if os.path.isdir("reference/.git"):
        run("git rm --cached -r reference")
        run("rm -rf .git/modules/reference")

    print("➕ Adding all files...")
    run("git add -f .")  # force includes previously ignored files

    print("✅ Committing...")
    run('git commit -m "Clean initial commit of IDEFIX codebase"')

    print("📤 Pushing to GitHub...")
    run("git push -u origin main")

    print("🎉 All done! Repo is live at: https://github.com/mlehmann86/IDEFIX-ml")

if __name__ == "__main__":
    main()
