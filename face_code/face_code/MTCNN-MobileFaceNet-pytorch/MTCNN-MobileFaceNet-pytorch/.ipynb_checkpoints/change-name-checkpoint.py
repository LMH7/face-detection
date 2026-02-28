input_file = 'dataset/test/test.txt'
output_file = 'dataset/test/output.txt'

with open(input_file, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_line = 'dataset/test/' + line.strip() + '\n'
    new_lines.append(new_line)

with open(output_file, 'w') as f:
    f.writelines(new_lines)