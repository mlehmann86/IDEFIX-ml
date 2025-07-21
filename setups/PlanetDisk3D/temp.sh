submit_idefix() {
    module purge
    unset CUDA CUDA_HOME LD_LIBRARY_PATH PATH
    export PATH=/usr/local/bin:/usr/bin:/bin

    if [ $# -lt 5 ]; then
        echo "Usage: submit_idefix <setup_name> <sim_name> <machine: kolong|gp|ngabul> <reconstruction: Linear|Parabolic> <mode: cpu|gpu> [continuation|debug] [restart_num]"
        return 1
    fi

    local setup=$1
    local simname=$2
    local machine=$3
    local reconstruction=$4
    local mode=$5
    local run_modifier=${6:-}
    local manual_restart_num=${7:-}

    local is_continuation=0
    local debug_mode=""
    if [ "$run_modifier" = "continuation" ]; then
        is_continuation=1
    elif [ "$run_modifier" = "debug" ]; then
        debug_mode="debug"
    fi
    local restart_flag=""

    local thishost
    thishost=$(hostname)

    if [ "$thishost" = "gp8" ]; then
        basedir="/theory/lts/mlehmann/idefix-mkl"
        echo "🔁 Detected gp8 — syncing idefix-mkl from kolong..."
        rsync -av --progress --exclude='outputs' kolong:/tiara/home/mlehmann/data/idefix-mkl/ "$basedir/"
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: rsync from kolong failed!"
            return 1
        fi
    else
        basedir="/tiara/home/mlehmann/data/idefix-mkl"
    fi
    
    cd "$basedir" || return 1
    
    local inputfile="idefix.ini"
    local sim_output_dir="$basedir/outputs/$setup/$simname"
    local setup_source_dir="setups/$setup"

    # For a new run, create the output directory and copy setup files for archival
    if [ "$is_continuation" -eq 0 ]; then
        echo "✨ STARTING NEW SIMULATION"
        mkdir -p "$sim_output_dir"
        cp "$setup_source_dir/setup.cpp" "$sim_output_dir/"
        cp "$setup_source_dir/definitions.hpp" "$sim_output_dir/"
        cp "$setup_source_dir/$inputfile" "$sim_output_dir/"
    else
        echo "🔄 CONTINUATION MODE DETECTED"
    fi

    # The .ini file used for the run is always in the base directory
    local live_ini_file="$basedir/$inputfile"
    
    # For a continuation, get the .ini from the simulation dir. For a new run, get it from the setup dir.
    if [ "$is_continuation" -eq 1 ]; then
        cp "$sim_output_dir/$inputfile" "$live_ini_file"
    else
        cp "$setup_source_dir/$inputfile" "$live_ini_file"
    fi

    # Machine-specific setup
    case $machine in
        kolong)
            if [ "$mode" = "gpu" ]; then
                module load cuda/11.7 intel/2022.1.0 openmpi/4.1.4
                kokkos_arch="-DKokkos_ARCH_AMPERE86=ON" # Corrected architecture
                nproc=8
            else # cpu
                module load gcc/10.3.0 openmpi/4.1.4
                kokkos_arch=""
                nproc=2
            fi
            export CC=mpicc CXX=mpicxx
            ;;
        gp)
            module purge
            thishost=$(hostname)
            if [ "$thishost" = "gp8" ]; then
                module load gcc/11.2.0 cuda/12.8 openmpi/4.1.4 cmake/3.22.1
            else
                module load gcc/9.1.0 cuda/11.3 openmpi/4.0.4 cmake/3.18.1
            fi
            export CC=mpicc CXX=mpicxx
            kokkos_arch="-DKokkos_ARCH_PASCAL60=ON"
            nproc=4
            ;;
        # ... other machines ...
    esac

    if [ "$debug_mode" = "debug" ]; then
        nproc=1
    fi

    # --- Always recompile ---
    echo "📄 Copying source files for compilation..."
    if [ "$is_continuation" -eq 1 ]; then
        # For continuation, compile with the files from the sim directory
        cp -v "$sim_output_dir/setup.cpp" src/setup.cpp
        cp -v "$sim_output_dir/definitions.hpp" src/definitions.hpp
    else
        # For new run, compile with files from the setup directory
        cp -v "$setup_source_dir/setup.cpp" src/setup.cpp
        cp -v "$setup_source_dir/definitions.hpp" src/definitions.hpp
    fi
    
    echo "🧹 Removing old build directory..."
    rm -rf build CMakeCache.txt CMakeFiles

    echo "🔧 Configuring and compiling IDEFIX..."
    # (The cmake command block remains the same as before)
    # ...
    cmake --build build --target idefix -j || { echo "⚠️ Build failed"; return 1; }
    echo "✅ Build complete: build/idefix"


    # --- Domain Decomposition Logic ---
    # This logic now correctly operates on the live .ini file in the base directory
    local ini_to_check="$live_ini_file"
    # (The domain decomposition logic block remains the same as your original)
    # ...
    
    # --- Restart Logic ---
    if [ "$is_continuation" -eq 1 ]; then
        # Auto-detect restart number from the simulation output directory
        if [ -n "$manual_restart_num" ]; then
            restart_n=$manual_restart_num
        else
            last_file=$(ls -1 "$sim_output_dir"/data.*.vtk 2>/dev/null | sed 's/.*data\.\([0-9]*\)\.vtk/\1/' | sort -n | tail -n 1)
            if [ -z "$last_file" ]; then echo "❌ No .vtk files found in $sim_output_dir"; return 1; fi
            restart_n=$((10#$last_file))
        fi
        restart_flag="-restart $restart_n"
        echo "✅ Restarting from file number: $restart_n"
    fi
    
    # --- Launch Simulation ---
    # The .out log file goes into the sim_output_dir
    local outfile="$sim_output_dir/$simname.out"
    local idefix_exec="$basedir/build/idefix"
    if [ "$mode" = "cpu" ]; then mpirun_extra="--mca pml ob1"; else mpirun_extra=""; fi
    if grep -qi "^X3-grid" "$ini_to_check"; then
        dec_arg="$(grep -i 'nx1' "$ini_to_check" | awk '{print $3}') $(grep -i 'nx2' "$ini_to_check" | awk '{print $3}') $(grep -i 'nx3' "$ini_to_check" | awk '{print $3}')"
    else
        dec_arg="$(grep -i 'nx1' "$ini_to_check" | awk '{print $3}') $(grep -i 'nx2' "$ini_to_check" | awk '{print $3}')"
    fi

    echo "🚀 Running simulation '$simname'..."
    echo "Executing: nohup mpirun $mpirun_extra -np "$nproc" $idefix_exec -dec $dec_arg $restart_flag > "$outfile" 2>&1 &"
    
    # The script is already in $basedir, so we run from here
    nohup mpirun $mpirun_extra -np "$nproc" "$idefix_exec" -dec $dec_arg $restart_flag > "$outfile" 2>&1 &
}
