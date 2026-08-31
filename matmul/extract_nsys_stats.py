import os

def extract_nsys_stats(prefix):
    stats = {}
    
    log_file = f'{prefix}_log.txt'
    rep_file = f'{prefix}.nsys-rep'
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'Kernel' in content:
            import re
            kernel_lines = re.findall(r'Kernel.*\n.*\n.*\n.*\n.*\n.*\n.*\n', content)
            for kl in kernel_lines[:3]:
                lines = kl.strip().split('\n')
                if len(lines) >= 5:
                    stats['kernel_name'] = lines[0].strip()
                    stats['time_ms'] = lines[2].strip() if 'Time' in lines[2] else ''
                    stats['gflops'] = lines[3].strip() if 'GFLOPS' in lines[3] else ''
                    break
    
    if os.path.exists(rep_file):
        stats['rep_exists'] = True
    else:
        stats['rep_exists'] = False
    
    return stats

v1_stats = extract_nsys_stats('matmul_v1_nsys')
v2_stats = extract_nsys_stats('matmul_v2_nsys')

print("=" * 70)
print("nsys Statistics Comparison - V1 vs V2")
print("=" * 70)
print()

print(f"{'Metric':<40} {'V1':<30} {'V2':<30}")
print("-" * 100)

print(f"{'Kernel Name':<40} {v1_stats.get('kernel_name', 'N/A'):<30} {v2_stats.get('kernel_name', 'N/A'):<30}")
print(f"{'Execution Time':<40} {v1_stats.get('time_ms', 'N/A'):<30} {v2_stats.get('time_ms', 'N/A'):<30}")
print(f"{'GFLOPS':<40} {v1_stats.get('gflops', 'N/A'):<30} {v2_stats.get('gflops', 'N/A'):<30}")
print(f"{'Report Exists':<40} {'Yes' if v1_stats.get('rep_exists') else 'No':<30} {'Yes' if v2_stats.get('rep_exists') else 'No':<30}")

print()
print("=" * 70)
print("How to Investigate Further")
print("=" * 70)
print()
print("1. Open nsys reports in nsys-ui:")
print("   nsys-ui matmul_v1_nsys.nsys-rep")
print("   nsys-ui matmul_v2_nsys.nsys-rep")
print()
print("2. Look for these metrics in nsys-ui:")
print("   - Kernel execution time breakdown")
print("   - Memory bandwidth usage")
print("   - SM occupancy")
print("   - Warp stall reasons")
print()
print("3. Compare these key metrics:")
print("   - Global memory read/write throughput")
print("   - Shared memory efficiency")
print("   - Warp utilization")
print("   - Instruction throughput")
print()
print("4. If you have NVIDIA Nsight Compute (ncu), run:")
print("   ncu --set full --optimization-report matmul_v1.exe")
print("   ncu --set full --optimization-report matmul_v2.exe")
print()
print("5. The most likely cause of V2 slowdown:")
print("   - Reduced SM occupancy due to increased shared memory")
print("   - Compiler generating less efficient code for padded arrays")
print("   - Bank conflict resolution adding overhead without benefit")
print()