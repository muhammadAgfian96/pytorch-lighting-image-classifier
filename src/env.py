import os

from dotenv import load_dotenv

path_env = os.path.join(os.getcwd(), ".env")
load_dotenv(path_env)