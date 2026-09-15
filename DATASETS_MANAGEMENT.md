# Dataset setup (local, one-time per machine)

The `datasets/` folder is **not included** in this repository — it's excluded
via `.gitignore` to avoid bloating the repo with ~15MB of files everyone
already has from the course. You need to add it manually, once, after cloning.

This is a one-time setup per machine. Pulling future updates from the repo
will **not** affect or erase your local `datasets/` folder. 

## Steps

1. **Clone the repo** (skip if already done):
```bash
   git clone https://github.com/RishitMahapatra/QEC-data-pipeline.git
   cd QEC-data-pipeline
```

2. **Locate your original `assignment-1.zip`** — the file you downloaded
   directly from the course page. Check your Downloads folder or wherever
   you saved it.

3. **Extract the `datasets/` folder** from that zip, and place it inside
   your cloned repo so the structure looks like:

   <p align= centre> <img width="275" height="203" alt="image" src="https://github.com/user-attachments/assets/81ebe72f-5feb-42ae-a727-afa030410c3e" /> </p>




5. **Verify the QASMBench source stayed as a zip file.** This is the one
   part that's easy to get wrong. Check:
```bash
   ls datasets/student-bundle/core/raw/source=qasmbench/
```
   You should see exactly:
   qasmbench-qec.zip

      If instead you see an **extracted folder** called `qasmbench-qec/`
   (containing files like `LICENSE`, `README.md`, `small/`, etc.), your zip
   tool auto-extracted it. Delete that folder and re-extract just the zip
   file itself without letting it unpack automatically.

5. **Run bootstrap:**
```bash
   make bootstrap
```

## If bootstrap fails with "Missing bundle object"

This means one of the three dataset zips isn't sitting where it should be,
or got extracted instead of staying zipped. Re-check step 4 above — this
exact error happened once already during setup and was caused by
`qasmbench-qec.zip` being auto-extracted into a folder instead of staying
as a `.zip`.
