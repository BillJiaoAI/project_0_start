import csv
import os

def parse_ncu_csv(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = {}
        for row in reader:
            if len(row) >= 3:
                metric = row[1].strip()
                value = row[2].strip()
                data[metric] = value
        return data

v1 = parse_ncu_csv('ncu_results.txt')
v2 = parse_ncu_csv('ncu_results_v2.txt')

print("=" * 70)
print("V1 vs V2 Performance Comparison")
print("=" * 70)
print()

metrics = [
    ('sm__warps_active.avg.pct_of_peak_sustained_active', 'SM Occupancy (%)'),
    ('sm__registers_per_thread.avg', 'Registers/Thread'),
    ('sm__shared_mem_per_block.avg', 'Shared Memory/Block (bytes)'),
    ('sm__inst_executed.avg', 'Instructions Executed'),
    ('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'Throughput (%)'),
]

print(f"{'Metric':<60} {'V1':<20} {'V2':<20} {'Diff':<10}")
print("-" * 110)

for metric, name in metrics:
    v1_val = v1.get(metric, 'N/A')
    v2_val = v2.get(metric, 'N/A')
    diff = ""
    try:
        if v1_val != 'N/A' and v2_val != 'N/A':
            v1_float = float(v1_val)
            v2_float = float(v2_val)
            if v1_float != 0:
                diff = f"{((v2_float - v1_float) / v1_float * 100):+.1f}%"
    except:
        pass
    print(f"{name:<60} {v1_val:<20} {v2_val:<20} {diff:<10}")

print()
print("=" * 70)
print("Key Findings")
print("=" * 70)

v1_occ = v1.get('sm__warps_active.avg.pct_of_peak_sustained_active', 'N/A')
v2_occ = v2.get('sm__warps_active.avg.pct_of_peak_sustained_active', 'N/A')

try:
    v1_occ_float = float(v1_occ) if v1_occ != 'N/A' else None
    v2_occ_float = float(v2_occ) if v2_occ != 'N/A' else None
    
    if v1_occ_float is not None and v2_occ_float is not None:
        if v2_occ_float < v1_occ_float - 5:
            print("✗ SM Occupancy dropped significantly!")
            print(f"  This is likely the cause of performance regression.")
            print(f"  Padding increased shared memory usage, reducing block count per SM.")
        else:
            print("✓ SM Occupancy is similar, not the cause.")
except:
    print("? Cannot determine occupancy difference.")

v1_smem = v1.get('sm__shared_mem_per_block.avg', 'N/A')
v2_smem = v2.get('sm__shared_mem_per_block.avg', 'N/A')

try:
    v1_smem_float = float(v1_smem) if v1_smem != 'N/A' else None
    v2_smem_float = float(v2_smem) if v2_smem != 'N/A' else None
    
    if v1_smem_float is not None and v2_smem_float is not None:
        increase = v2_smem_float - v1_smem_float
        print(f"\nShared Memory increase: {increase:.0f} bytes")
        print(f"  V1: {v1_smem_float:.0f} bytes")
        print(f"  V2: {v2_smem_float:.0f} bytes")
        
        sm_max_smem = 102400
        v1_blocks = sm_max_smem / v1_smem_float
        v2_blocks = sm_max_smem / v2_smem_float
        print(f"\nTheoretical blocks per SM:")
        print(f"  V1: {v1_blocks:.1f}")
        print(f"  V2: {v2_blocks:.1f}")
        print(f"  Difference: {v2_blocks - v1_blocks:.1f} blocks")
        
        if v1_blocks - v2_blocks > 1:
            print("\n⚠  WARNING: SM can hold significantly fewer blocks with padding!")
        else:
            print("\n✓ Shared memory increase is minimal, unlikely to cause regression.")
except:
    print("\n? Cannot compare shared memory usage.")

v1_inst = v1.get('sm__inst_executed.avg', 'N/A')
v2_inst = v2.get('sm__inst_executed.avg', 'N/A')

try:
    v1_inst_float = float(v1_inst) if v1_inst != 'N/A' else None
    v2_inst_float = float(v2_inst) if v2_inst != 'N/A' else None
    
    if v1_inst_float is not None and v2_inst_float is not None:
        increase = ((v2_inst_float - v1_inst_float) / v1_inst_float * 100)
        print(f"\nInstructions executed change: {increase:+.1f}%")
        if increase > 5:
            print("✗ Instructions increased significantly!")
            print("  Padding may have affected compiler optimization.")
        else:
            print("✓ Instruction count is similar.")
except:
    print("\n? Cannot compare instruction count.")

print()
print("=" * 70)
print("Recommendations")
print("=" * 70)
print("1. Check nsys profile for kernel execution time breakdown")
print("2. Use ncu --set full for detailed optimization report")
print("3. Compare PTX assembly for instruction differences")
print("4. Consider removing padding if occupancy drops")
print("5. Try using __ldg() for read-only data to bypass shared memory")
print()
print("To investigate further:")
print("  nsys-ui matmul_v1_nsys.nsys-rep")
print("  nsys-ui matmul_v2_nsys.nsys-rep")
print()