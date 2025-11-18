from pathlib import Path
path = Path(r'c:/dev/Powershell/amp/src/native/nodes/fft_division/fft_division_nodes.inc')
text = path.read_text()
needle = 'static int fftdiv_execute_block('
start = text.index(needle)
brace = text.index('{', start)
depth = 1
pos = brace + 1
while pos < len(text) and depth > 0:
    ch = text[pos]
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
    pos += 1
if depth != 0:
    raise RuntimeError('Unbalanced braces removing function')
end = pos  # position after closing brace
new_text = text[:start] + text[end:]
path.write_text(new_text)
print('Removed fftdiv_execute_block, new length', len(new_text))
