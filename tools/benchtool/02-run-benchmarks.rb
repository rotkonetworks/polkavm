#!/usr/bin/ruby

require "shellwords"
require "fileutils"

$errors_are_fatal = true

def sudo_write(contents, path)
    system "echo #{contents.to_s.shellescape} | sudo tee #{path.shellescape} > /dev/null"
    if $?.exitstatus != 0
        STDERR.puts "ERROR: failed to write to '#{path}'"
        if $errors_are_fatal
            STDERR.puts "Aborting..."
            exit 1
        end
    end
end

def sudo_run(command)
    system "sudo #{command}"
    if $?.exitstatus != 0
        STDERR.puts "ERROR: command failed: #{command}"
        if $errors_are_fatal
            STDERR.puts "Aborting..."
            exit 1
        end
    end
end

def ensure_command_installed(command)
    if !system("command -v #{command.shellescape} > /dev/null 2>&1")
        STDERR.puts "ERROR: Required command '#{command}' is not installed or not found in PATH."
        STDERR.puts "Please install it to continue."
        if command == "schedtool"
            STDERR.puts "  Ensure that 'schedtool' is installed."
        end
        if $errors_are_fatal
            STDERR.puts "Aborting..."
            exit 1
        end
        return false
    end
    STDERR.puts "Checked: Command '#{command}' is available."
    return true
end

BENCHMARK_KINDS = [
    "runtime",
    "compilation",
    "oneshot",
]

BENCHMARK_PROGRAMS = [
    "pinky",
    "prime-sieve",
    "minimal",
]

BENCHMARK_VMS = [
    "ckbvm_asm",
    "ckbvm_non_asm",
    "native",
    "polkavm32_compiler_no_gas",
    "polkavm32_compiler_async_gas",
    "polkavm32_compiler_sync_gas",
    "polkavm32_interpreter",
    "polkavm64_compiler_no_gas",
    "polkavm64_compiler_async_gas",
    "polkavm64_compiler_sync_gas",
    "polkavm64_interpreter",
    "solana_rbpf",
    "wasm3",
    "wasmer",
    "wasmi_eager_checked",
    "wasmi_eager_unchecked",
    "wasmi_lazy_checked",
    "wasmi_lazy_unchecked",
    "wasmi_lazy_translation_checked",
    "wasmi_lazy_translation_unchecked",
    "wasmtime_cranelift_default",
    "wasmtime_cranelift_with_fuel",
    "wasmtime_cranelift_with_epoch",
    "wasmtime_winch",
    "wazero",
]

if File.exist? "target/criterion"
    unless ARGV.include? "--keep-old-results"
        STDERR.puts "ERROR: 'target/criterion' directory exists! Either delete it or pass '--keep-old-results'"
        exit 1
    end
end

$num_cpus = IO.popen("getconf _NPROCESSORS_ONLN").read.strip.to_i
raise "ERROR: Failed to get CPU count or count is zero." if $num_cpus <= 0
$all_cpus_range = "0-#{$num_cpus - 1}"
$all_cpus_hex_mask = ($num_cpus == 0) ? "0" : ("%x" % ((1 << $num_cpus) - 1))

$bench_cpu_indices = [1, 2]
if $num_cpus < 3 && ($bench_cpu_indices.include?(1) || $bench_cpu_indices.include?(2))
    raise "ERROR: Script expects at least 3 CPUs to use CPUs 1 & 2 for benchmark. Found #{$num_cpus}."
end
$bench_cpu_indices.each do |cpu_idx|
    raise "ERROR: Benchmark CPU index #{cpu_idx} out of range (max #{$num_cpus - 1})." if cpu_idx >= $num_cpus
end

def to_cpuset_string(indices_array)
    return "" if indices_array.empty?
    sorted_indices = indices_array.uniq.sort
    return "" if sorted_indices.empty?
    ranges = []
    current_range_start = sorted_indices.first
    current_range_end = sorted_indices.first
    (1...sorted_indices.length).each do |i|
        idx = sorted_indices[i]
        if idx == current_range_end + 1
            current_range_end = idx
        else
            ranges << (current_range_start == current_range_end ? "#{current_range_start}" : "#{current_range_start}-#{current_range_end}")
            current_range_start = idx
            current_range_end = idx
        end
    end
    ranges << (current_range_start == current_range_end ? "#{current_range_start}" : "#{current_range_start}-#{current_range_end}")
    ranges.join(",")
end

$bench_cpus_str = to_cpuset_string($bench_cpu_indices)
other_cpu_indices_arr = (0...$num_cpus).to_a - $bench_cpu_indices
$other_cpus_str = to_cpuset_string(other_cpu_indices_arr)
raise "ERROR: Benchmark CPUs and Other CPUs configuration invalid." if $bench_cpus_str.empty? || ($other_cpus_str.empty? && $num_cpus > $bench_cpu_indices.length)


system "cargo build --release --features ckb-vm"
raise "failed to build benchtool" unless $?.exitstatus == 0

FileUtils.mkdir_p "target/criterion"

ensure_command_installed("schedtool")

def read_sys_file(path)
    File.exist?(path) ? File.read(path).strip : nil
end

original_sched_rt = read_sys_file("/proc/sys/kernel/sched_rt_runtime_us")
original_watchdog = read_sys_file("/proc/sys/kernel/watchdog")
original_stat_interval = read_sys_file("/proc/sys/vm/stat_interval")
original_boost = read_sys_file("/sys/devices/system/cpu/cpufreq/boost")

original_wq_mask = read_sys_file("/sys/devices/virtual/workqueue/cpumask")
original_wb_mask = read_sys_file("/sys/bus/workqueue/devices/writeback/cpumask")
original_irq_affinity = read_sys_file("/proc/irq/default_smp_affinity")

$original_governors = {}
(0...$num_cpus).each do |i|
    gov_file = "/sys/devices/system/cpu/cpu#{i}/cpufreq/scaling_governor"
    $original_governors[i] = read_sys_file(gov_file) if File.exist?(gov_file)
end

begin
    system "sync"

    STDERR.puts "Disabling turbo boost..."
    sudo_write "0", "/sys/devices/system/cpu/cpufreq/boost" if original_boost

    STDERR.puts "Applying misc. tweaks..."
    sudo_write "-1", "/proc/sys/kernel/sched_rt_runtime_us" if original_sched_rt
    sudo_write "0", "/proc/sys/kernel/watchdog" if original_watchdog
    sudo_write "1000", "/proc/sys/vm/stat_interval" if original_stat_interval

    STDERR.puts "Tweaking CPU masks..."
    cpu0_is_other = !$bench_cpu_indices.include?(0) && $num_cpus > 0
    if cpu0_is_other
        sudo_write "1", "/sys/devices/virtual/workqueue/cpumask" if original_wq_mask
        sudo_write "1", "/sys/bus/workqueue/devices/writeback/cpumask" if original_wb_mask
        sudo_write "1", "/proc/irq/default_smp_affinity" if original_irq_affinity
    end

    $bench_cpu_indices.each do |cpu_idx|
        STDERR.puts "Changing the scaling governor to 'performance' for cpu#{cpu_idx}..."
        gov_file = "/sys/devices/system/cpu/cpu#{cpu_idx}/cpufreq/scaling_governor"
        sudo_write "performance", gov_file if File.exist?(gov_file)
    end

    STDERR.puts "Setting up cgroups..."
    sudo_run "mkdir -p /sys/fs/cgroup/benchtool"
    sudo_write "+cpuset", "/sys/fs/cgroup/benchtool/cgroup.subtree_control"
    sudo_write $bench_cpus_str, "/sys/fs/cgroup/benchtool/cpuset.cpus"

    unless $other_cpus_str.empty?
        sudo_write $other_cpus_str, "/sys/fs/cgroup/user.slice/cpuset.cpus" if File.exist?("/sys/fs/cgroup/user.slice/cpuset.cpus")
        sudo_write $other_cpus_str, "/sys/fs/cgroup/system.slice/cpuset.cpus" if File.exist?("/sys/fs/cgroup/system.slice/cpuset.cpus")
    end

    STDERR.puts "Launching child process..."
    rx, tx = IO.pipe
    child = Kernel.fork do
        tx.close
        rx.read # Wait for the parent process to add us to the cgroup.
        rx.close

        STDERR.puts "Running benchmarks..."
        cpu = read_sys_file("/proc/cpuinfo")&.scan(/model name\s*:\s*(.+)/)&.first&.first || "Unknown"
        commit = `git rev-parse HEAD 2>/dev/null`.strip
        commit = "unknown" if commit.empty?
        File.write("target/criterion/cpu.txt", cpu)
        File.write("target/criterion/commit.txt", commit)
        File.write("target/criterion/platform.txt", RUBY_PLATFORM)

        BENCHMARK_KINDS.each do |kind|
            BENCHMARK_PROGRAMS.each do |program|
                BENCHMARK_VMS.each do |vm|
                    next if ARGV.include?("--keep-old-results") && File.exist?("target/criterion/#{kind}_#{program}/#{vm}/new/estimates.json")
                    system "target/release/benchtool criterion #{kind}/#{program}/#{vm}"
                end
            end
        end
        exit 0
    end

    STDERR.puts "Adding child to cgroup and setting its priority..."
    sudo_write child, "/sys/fs/cgroup/benchtool/cgroup.procs"
    sudo_run "schedtool -F -p 99 -n -20 #{child}"

    rx.close
    tx.close
    Process.wait child
    raise "Benchmark child process failed!" if $?.exitstatus != 0

    ensure
        $errors_are_fatal = false

        STDERR.puts "Restoring turbo boost..."
        sudo_write (original_boost == "0" ? "0" : "1"), "/sys/devices/system/cpu/cpufreq/boost" if original_boost

        $original_governors.each do |cpu_idx, original_gov|
            STDERR.puts "Restoring the scaling governor to '#{original_gov}' for cpu#{cpu_idx}..."
            sudo_write original_gov, "/sys/devices/system/cpu/cpu#{cpu_idx}/cpufreq/scaling_governor" if original_gov
        end

        STDERR.puts "Restoring cgroups..."
        sudo_write $all_cpus_range, "/sys/fs/cgroup/user.slice/cpuset.cpus" if File.exist?("/sys/fs/cgroup/user.slice/cpuset.cpus")
        sudo_write $all_cpus_range, "/sys/fs/cgroup/system.slice/cpuset.cpus" if File.exist?("/sys/fs/cgroup/system.slice/cpuset.cpus")
        sudo_run "rmdir /sys/fs/cgroup/benchtool"

        STDERR.puts "Restoring misc. tweaks..."
        sudo_write original_wq_mask, "/sys/devices/virtual/workqueue/cpumask" if original_wq_mask
        sudo_write original_wb_mask, "/sys/bus/workqueue/devices/writeback/cpumask" if original_wb_mask
        sudo_write original_irq_affinity, "/proc/irq/default_smp_affinity" if original_irq_affinity

        sudo_write original_sched_rt, "/proc/sys/kernel/sched_rt_runtime_us" if original_sched_rt
        sudo_write original_watchdog, "/proc/sys/kernel/watchdog" if original_watchdog
        sudo_write original_stat_interval, "/proc/sys/vm/stat_interval" if original_stat_interval

        STDERR.puts "Original state restored!"
end
