import os
import subprocess

REPO_PATH = "/tiara/home/mlehmann/data/idefix-mkl"
REMOTE_URL = "git@github.com:mlehmann86/IDEFIX-ml.git"

def run(cmd):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    os.chdir(REPO_PATH)

    print("🧼 Init Git")
    run("git init")
    run(f"git remote add origin {REMOTE_URL}")
    run("git branch -M main")

    print("📄 Writing .gitignore")
    with open(".gitignore", "w") as f:
        f.write("outputs/\nreference/\n*.out\n*.err\n*.dat\n*.vtk\n*.o\n*.mod\nbuild/\ncmake-build*/\n")

    print("➕ Adding actual files")
    run("git add src/ setups/ CMakeLists.txt")

    print("✅ Committing and pushing")
    run('git commit -m "Clean initial IDEFIX commit"')
    run("git push -u origin main")

    print("🎉 Done!")

if __name__ == "__main__":
    main()
