#!/usr/bin/env python
"""
Script to regenerate gRPC code from ml_server.proto.

Usage:
    python generate_grpc.py

This regenerates the protobuf and gRPC Python code from the LanguageTool
ml_server.proto definition.
"""

import subprocess
import sys
from pathlib import Path

def regenerate_grpc():
    """Regenerate gRPC code from ml_server.proto."""
    src_dir = Path(__file__).parent
    grpc_gen_dir = src_dir / "grpc_gen"
    proto_file = src_dir / "ml_server.proto"
    
    if not proto_file.exists():
        print(f"Error: {proto_file} not found")
        return False
    
    if not grpc_gen_dir.exists():
        grpc_gen_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {grpc_gen_dir}")
    
    print(f"Regenerating gRPC code from {proto_file}...")
    print(f"Output directory: {grpc_gen_dir}")
    
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{src_dir}",
        f"--python_out={grpc_gen_dir}",
        f"--pyi_out={grpc_gen_dir}",
        f"--grpc_python_out={grpc_gen_dir}",
        str(proto_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ gRPC code regenerated successfully")
                
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error regenerating gRPC code:")
        print(f"Command: {' '.join(cmd)}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("✗ grpc_tools.protoc not found. Install with: pip install grpcio-tools")
        return False

if __name__ == "__main__":
    success = regenerate_grpc()
    sys.exit(0 if success else 1)
