"""
Utility functions for the project
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class Utils:
    @staticmethod
    def save_json(data, filepath):
        """Save data to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved to {filepath}")
    
    @staticmethod
    def load_json(filepath):
        """Load JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_csv(df, filepath):
        """Save DataFrame to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Saved to {filepath}")
    
    @staticmethod
    def get_file_size(filepath):
        """Get file size in KB/MB"""
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
    
    @staticmethod
    def list_files(directory):
        """List all files in directory"""
        if not os.path.exists(directory):
            return []
        files = []
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)
            size = Utils.get_file_size(filepath)
            files.append({'name': file, 'size': size})
        return files
    
    @staticmethod
    def print_directory_structure(directory, indent=0):
        """Print directory tree structure"""
        if not os.path.exists(directory):
            print(f"{' ' * indent}❌ {directory} not found")
            return
        
        print(f"{' ' * indent}📁 {os.path.basename(directory)}/")
        indent += 4
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                Utils.print_directory_structure(item_path, indent)
            else:
                size = Utils.get_file_size(item_path)
                print(f"{' ' * indent}📄 {item} ({size})")
    
    @staticmethod
    def create_timestamp():
        """Create timestamp string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def validate_dataframe(df, required_columns):
        """Validate if DataFrame has required columns"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            return False
        return True