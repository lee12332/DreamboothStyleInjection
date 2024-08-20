import os
import subprocess

script_path = './loop.sh'


if not os.path.exists(script_path):
    print("Script does not exist.")
    exit(1)


# if not os.access(script_path, os.X_OK):
#     print("Script does not have execute permissions. Adding execute permissions...")
#     os.chmod(script_path, os.stat(script_path).st_mode | 0o111)
# result = subprocess.run(script_path, text=True, capture_output=True)


loop_return=os.system('bash loop.sh')
print(loop_return)

# print("STDOUT:", result.stdout)
# print("STDERR:", result.stderr)