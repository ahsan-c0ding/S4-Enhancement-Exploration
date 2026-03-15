import subprocess
import sys
import os
import glob

def run_automated_tests():
    print("Starting Automated Layer-by-Layer Validation")

    print("Compiling test_app in ../c...")
    compile_process = subprocess.run(["make", "test_app"], cwd="../c", capture_output=True, text=True)
    if compile_process.returncode != 0:
        print(" Compilation Failed!")
        sys.exit(1)

    test_data_dir = "../test_data"
    img_files = sorted(glob.glob(os.path.join(test_data_dir, "sample_*_img.bin")))
    
    passed = 0
    failed = 0
    total_samples = len(img_files)

    for i, img_path in enumerate(img_files):
        # Extract just the prefix (e.g., "../test_data/sample_0")
        prefix = img_path.replace("_img.bin", "")
        
        print(f"\nEvaluating Sample {i+1}/{total_samples}: {os.path.basename(prefix)}...")
        
        # Pass the prefix to the C app
        test_process = subprocess.run(
            ["./test_app", prefix], 
            cwd="../c", capture_output=True, text=True
        )
        
        output = test_process.stdout
        print(output)

        if "PASSED" in output:
            passed += 1
        else:
            failed += 1

    print("Aggregate Test Results:")
    print(f"Total Samples: {total_samples}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0: sys.exit(1)
    else: sys.exit(0)

if __name__ == "__main__":
    run_automated_tests()
