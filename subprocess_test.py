# %%
from os import close
import subprocess

id = str(1)

subprocess.call(['rm', id])
subprocess.call(['mkdir', id])

atoms2 = molecule('CH3')
io.write('ch3.xyz', atoms2)

subprocess.call(['mv', 'ch3.xyz', './' + id])

process = subprocess.Popen(['xtb', 'ch3.xyz', '--gfn2', '--verbose'], cwd='./' + id, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

output = open(id + "/output", "w")
print(stdout.decode(), file=output)
print(stderr.decode(), file=output)

output.close()