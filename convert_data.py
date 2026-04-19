import struct
import os

def bin_to_asm(bin_file, asm_file, label, is_append=False):
    mode = 'a' if is_append else 'w'
    with open(bin_file, 'rb') as f_in, open(asm_file, mode) as f_out:
        if not is_append:
            f_out.write(".section .data\n")
            f_out.write(".align 4\n\n")

        f_out.write(f".global {label}\n")
        f_out.write(f"{label}:\n")

        while True:
            chunk = f_in.read(4)
            if not chunk or len(chunk) < 4:
                break
            val = struct.unpack('<I', chunk)[0]
            f_out.write(f"    .word 0x{val:08x}\n")
        f_out.write("\n")

if __name__ == "__main__":
    weights_path = "./model_params/model_weights.bin"
    image_path = "./test_data/sample_0_img.bin"
    out_path = "data.s"

    print(f"Converting {weights_path}...")
    bin_to_asm(weights_path, out_path, "weights_data", is_append=False)
    
    print(f"Converting {image_path}...")
    bin_to_asm(image_path, out_path, "image_data", is_append=True)
    
    print(f"Successfully generated {out_path}!")
