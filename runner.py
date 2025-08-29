import subprocess
import time
import sys

def run_main():
    try:
        # Run your main.py as a subprocess using the same interpreter
        subprocess.run(["python3", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        run_main()  # recursive retry

if __name__ == "__main__":
    run_main()
