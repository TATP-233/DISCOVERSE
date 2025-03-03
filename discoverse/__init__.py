import os

DISCOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.getenv("DISCOVERSE_ASSERT_DIR"):
    DISCOVERSE_ASSERT_DIR = os.getenv("DISCOVERSE_ASSERT_DIR")
    print(f'>>> get env "DISCOVERSE_ASSERT_DIR": {DISCOVERSE_ASSERT_DIR}')
else:
    DISCOVERSE_ASSERT_DIR = os.path.join(DISCOVERSE_ROOT_DIR, "models")
