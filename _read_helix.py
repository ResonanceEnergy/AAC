with open('shared/ax_helix_integration.py', 'rb') as f:
    data = f.read()
lines = data.split(b'\n')
print(f'TOTAL_LINES: {len(lines)}')
for i in range(95, min(len(lines), 108)):
    print(f'L{i+1}: {lines[i]!r}')
