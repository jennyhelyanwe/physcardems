# Getting started: your first heart simulation

This tutorial takes you from an empty laptop to a 3D simulation of a beating left ventricle.
You don't need to know git, Docker or FEniCSx beforehand. Follow the steps in order, and copy
the commands exactly. Allow about an hour for steps 1 to 4 (mostly downloads), then one to two
hours for the simulation itself, which runs on its own.

You need a laptop with at least 8 GB of memory and about 10 GB of free disk space, and an internet
connection for the installation.

## How the pieces fit together

- **Git** downloads the code and keeps it up to date. You will use two git commands: `git clone`
  (once) and `git pull` (to get updates).
- **Docker** runs a ready-made Linux environment, called a *container*, with all the scientific
  software installed. The simulation runs inside it, so it behaves the same on Mac and Windows.
- **VS Code** is a text editor for reading the code and editing settings files.
- **ParaView** displays the 3D results.

You will use two kinds of terminal (a window where you type commands):

| | What the prompt looks like | Used for |
|---|---|---|
| Laptop terminal | `yourname@laptop ~ %` (Mac) or `yourname@laptop:~$` (Windows, Ubuntu) | git, starting Docker, opening files |
| Container terminal | `root@3f2a1b...:/home/shared#` | running simulations |

If a command fails, first check that you're in the right one.

## Step 1: install the tools

### Mac

1. **Git:** open the Terminal app (press Cmd+Space, type *Terminal*), type `git --version` and
   press Enter. If git isn't installed, macOS offers to install the developer tools; accept, and
   wait for it to finish.
2. **Docker Desktop:** download it from https://www.docker.com/products/docker-desktop/. Choose
   *Apple Silicon* or *Intel* to match your Mac (Apple menu > About This Mac: "Chip: Apple M..."
   means Apple Silicon). Install and open it, accept the terms and skip the sign-in. Then go to
   Settings > Resources and set Memory to at least 6 GB.
3. **VS Code:** https://code.visualstudio.com/
4. **ParaView:** https://www.paraview.org/download/ (the latest release).

### Windows

Windows runs the Linux tools through *WSL* (Windows Subsystem for Linux). Docker needs it anyway.

1. **WSL with Ubuntu:** right-click the Start button, choose *Terminal (Admin)* or *PowerShell
   (Admin)*, and run:
```
   wsl --install
```
   Restart the laptop when asked. After the restart an Ubuntu window opens and asks you to choose
   a username and password (the password won't show as you type; that's normal). Remember them.
   From now on, **"laptop terminal" means this Ubuntu window** (find it later by typing *Ubuntu*
   in the Start menu). Git is already installed in it.
2. **Docker Desktop:** download it from https://www.docker.com/products/docker-desktop/, install
   it with the default options (keep "Use WSL 2" ticked), and open it. Accept the terms and skip
   the sign-in. In Settings > Resources > WSL integration, make sure Ubuntu is switched on.
3. **VS Code:** https://code.visualstudio.com/. Then, in VS Code, open Extensions (the four squares
   on the left) and install the **WSL** extension.
4. **ParaView:** https://www.paraview.org/download/ (the Windows installer, latest release).

## Step 2: get the code

In the laptop terminal:

```bash
cd ~
git clone https://github.com/jennyhelyanwe/physcardems.git
cd physcardems
```

This creates a folder `physcardems` in your home folder with all the code.

To open it in VS Code:
- **Mac:** open VS Code, then File > Open Folder and choose `physcardems` in your home folder.
- **Windows:** in the Ubuntu window, from the `physcardems` folder, type `code .` (the first time,
  this sets up the WSL connection for you).

## Step 3: start the container and install the software (once)

Make sure Docker Desktop is open (on a Mac, the whale icon appears in the menu bar). Then, in the
laptop terminal, in the `physcardems` folder:

```bash
bash docker/start_docker.sh
```

The first time, this downloads the FEniCSx environment (a few GB, so it takes a while). When it
finishes, the prompt changes to `root@...:/home/shared#`: you are now in the container terminal.
`/home/shared` inside the container *is* your `physcardems` folder, so files you change on your
laptop appear in the container, and simulation results appear on your laptop.

Now install the cardiac modelling packages (once; about 10 to 15 minutes):

```bash
bash docker/setup.sh
```

It ends by printing the version numbers of `pulse`, `beat` and `gotranx`. If it does, the
installation worked.

To leave the container, type `exit`. To come back later, run `bash docker/start_docker.sh` again
from the `physcardems` folder; it reopens the same container, with everything still installed.

## Step 4: your first simulation

The model's settings are in a config file, `configs/lv_ellipsoid/baseline.toml`. Don't edit that
file. Copy it into your own folder and edit the copy:

```bash
mkdir -p configs/my_runs runs
cp configs/lv_ellipsoid/baseline.toml configs/my_runs/my_first_run.toml
```

Open `configs/my_runs/my_first_run.toml` in VS Code. The baseline simulates two heartbeats; for a
first run, one is enough. Find the line

```toml
t_end_ms = 1600.0
```

change it to `t_end_ms = 800.0` and save. While you're there, read the comments: the file is
organised by the things you're likely to change (ion channels, active tension, stiffness,
circulation).

Then, in the container terminal:

```bash
python3 scripts/run_em_lvonly.py configs/my_runs/my_first_run.toml 2>&1 | tee runs/my_first_run.out
```

The first few minutes are set-up (generating and compiling code). Then a line appears for each
2 ms of simulated time, for example:

```
t= 200.0 ms | Ta max  282.2 kPa, lambda [...] | LV EJECTION V=  81.6 mL P= 12.27 kPa ...
```

It shows the time, the peak active tension, the cardiac cycle phase, and the LV volume (`V`) and
pressure (`P`). The run finishes at `t= 800.0 ms`, after one to two hours.

While it runs:
- Keep the terminal window open and the laptop plugged in and awake (closing the window stops
  the simulation). On a Mac you can prevent sleep by running `caffeinate -i` in a second laptop
  terminal; on Windows, set Settings > System > Power to never sleep while plugged in.
- To look around in the container without disturbing the run, open a second laptop terminal and
  run `docker exec -it physcardems bash`.

## Step 5: look at the results

Results are in `runs/lv_ellipsoid/my_first_run/`. The figures are refreshed during the run, so you
can look before it finishes.

**summary.png** shows pressure and volume over time, the pressure-volume loop with the ejection
fraction, and the pseudo-ECG. Open it from VS Code's file list, or from the laptop terminal:
- Mac: `open runs/lv_ellipsoid/my_first_run/summary.png`
- Windows: `explorer.exe runs/lv_ellipsoid/my_first_run` (opens the folder in File Explorer)

You should see an ejection fraction of about 63% and a peak pressure of about 13.7 kPa.

**The 3D deformation, in ParaView:**
1. File > Open, go to `runs/lv_ellipsoid/my_first_run/`, select the folder `simulation.bp`
   and click OK. (On Windows, your Ubuntu files are under *Linux > Ubuntu > home > yourname* in
   the file browser.) If ParaView asks which reader to use, choose **ADIOS2VTXReader**.
2. Click **Apply**.
3. Filters > Alphabetical > **Warp By Vector**, with Vectors set to `u`, then **Apply**. This
   moves the mesh by the computed displacement.
4. Colour by `Ta_kPa` (active tension) or `v_mV` (voltage) using the drop-down in the toolbar.
5. Press the green Play button to watch the beat.

## Step 6: change something

Copy your config, change one parameter, and run again. For example, to block the L-type calcium
channel by half:

```bash
cp configs/my_runs/my_first_run.toml configs/my_runs/half_ical.toml
```

In `half_ical.toml`, change `PCa_b = 2.0` to `PCa_b = 1.0`, then run it the same way as in step 4.
Before it finishes, write down what you expect to happen to the ejection fraction and the PV loop,
and why. Because this changes an ion channel, the run starts by pacing single cells to a new
steady state (a few extra minutes, once per parameter set).

Each run's results go to `runs/lv_ellipsoid/<config file name>/`. To compare two runs:

```bash
python3 scripts/compare_meshes_lv.py runs/lv_ellipsoid/my_first_run runs/lv_ellipsoid/half_ical --out runs/compare.png
```

To understand the model and what each parameter does, read
[`cases/lv_ellipsoid/README.md`](../cases/lv_ellipsoid/README.md) next.

## Getting updates

When you're told the code has been updated, in the laptop terminal, in the `physcardems` folder:

```bash
git pull
```

Your own configs (in `configs/my_runs/`) and results (in `runs/`) are not touched by this.

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` or `Cannot connect to the Docker daemon` | Docker Desktop isn't running. Open it, wait for it to start, and try again. On Windows, also check Settings > Resources > WSL integration. |
| `python3 scripts/...` says `No such file or directory` | You're in the laptop terminal or the wrong folder. Start the container (step 3); the prompt should end in `/home/shared#`. |
| `ModuleNotFoundError: No module named 'physcardems'` (or `pulse`, `beat`) | The installation didn't run in this container. Run `bash docker/setup.sh` again. |
| `FileNotFoundError` mentioning `case.toml` | Your config isn't in a folder directly under `configs/` (for example `configs/my_runs/`). The config finds the anatomy relative to its own location. |
| The run stops with `Killed` | The container ran out of memory. On a Mac, raise Memory in Docker Desktop (Settings > Resources) and close other programs. |
| The run stopped when the laptop slept or the window closed | Start it again; see the tips in step 4. |
| `git pull` says *Your local changes would be overwritten* | You edited a file that belongs to the repository. Run `git stash`, then `git pull`. Next time, make your changes in copies in `configs/my_runs/`. |

If you're still stuck, send an email with:
1. your laptop type (Mac or Windows),
2. the exact command you ran,
3. the last 30 lines of the output (for a simulation, `tail -n 30 runs/my_first_run.out`).