import os

def count_instructions(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    instructions = {
        'ld.shared': 0, 'st.shared': 0,
        'ld.global': 0, 'st.global': 0,
        'fadd': 0, 'fmul': 0, 'fma': 0,
        'mov': 0, 'sync': 0, 'bra': 0,
        'total': 0,
    }
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        
        if '.shared.' in line:
            if 'ld.shared' in line:
                instructions['ld.shared'] += 1
            elif 'st.shared' in line:
                instructions['st.shared'] += 1
        
        if '.global.' in line:
            if 'ld.global' in line:
                instructions['ld.global'] += 1
            elif 'st.global' in line:
                instructions['st.global'] += 1
        
        if 'fadd.' in line and not 'fadd_rn' in line:
            instructions['fadd'] += 1
        elif 'fmul.' in line:
            instructions['fmul'] += 1
        elif 'fma.' in line:
            instructions['fma'] += 1
        
        if 'mov.' in line:
            instructions['mov'] += 1
        elif 'syncthreads' in line:
            instructions['sync'] += 1
        elif 'bra ' in line:
            instructions['bra'] += 1
        
        instructions['total'] += 1
    
    return instructions

v1_inst = count_instructions('optimized_v1/matmul_v1.ptx')
v2_inst = count_instructions('optimized_v2/matmul_v2.ptx')

print("=" * 70)
print("PTX Instruction Comparison - V1 vs V2")
print("=" * 70)
print()

print(f"{'Instruction':<20} {'V1':<10} {'V2':<10} {'Diff':<10}")
print("-" * 50)

for key in ['total', 'ld.shared', 'st.shared', 'ld.global', 'st.global', 
            'fadd', 'fmul', 'fma', 'mov', 'sync', 'bra']:
    v1_val = v1_inst.get(key, 0)
    v2_val = v2_inst.get(key, 0)
    diff = f"{((v2_val - v1_val) / max(v1_val, 1) * 100):+.1f}%" if v1_val > 0 else "N/A"
    print(f"{key:<20} {v1_val:<10} {v2_val:<10} {diff:<10}")

print()
print("=" * 70)
print("Analysis")
print("=" * 70)

v1_total = v1_inst.get('total', 0)
v2_total = v2_inst.get('total', 0)

if v2_total > v1_total * 1.1:
    print("✗ V2 has significantly more instructions!")
    print("  This could explain the performance drop.")
else:
    print("✓ Instruction counts are similar.")

v1_shared = v1_inst.get('ld.shared', 0) + v1_inst.get('st.shared', 0)
v2_shared = v2_inst.get('ld.shared', 0) + v2_inst.get('st.shared', 0)

if v2_shared > v1_shared:
    print(f"\n✗ V2 has {v2_shared - v1_shared} more shared memory accesses")
else:
    print(f"\n✓ Shared memory accesses: V1={v1_shared}, V2={v2_shared}")

print()
print("Key Insight:")
print("If V2 has fewer instructions but runs slower,")
print("the issue is likely NOT in instruction count.")
print("Check nsys for memory bandwidth and SM occupancy.")
print()