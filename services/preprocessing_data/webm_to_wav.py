import os
import os
import subprocess
from tqdm import tqdm
import shutil


def find_ffmpeg():
    """Try to find ffmpeg binary path"""
    # Try to find ffmpeg in common locations
    search_paths = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/home/myl15/.local/bin/ffmpeg",
        "/home/myl15/.conda/envs/seamless/bin/ffmpeg",
        # Try to find the static version from the previous logs
        "/tmp/ffmpeg-static/ffmpeg"
    ]
    
    for path in search_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    # Try to find it in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
        
    return None

def convert_webm_to_wav(input_file, output_file, target_sr=16000):
    """Convert webm to wav using direct ffmpeg subprocess call"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Find ffmpeg path
        ffmpeg_path = find_ffmpeg()
        
        # If ffmpeg not found, try to use the one found by moviepy
        if not ffmpeg_path:
            # We know moviepy was able to find it earlier, so let's try a different approach
            #print("FFmpeg not found in standard locations, trying to use moviepy's version...")
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            #print(f"Found FFmpeg at: {ffmpeg_path}")
        
        if not ffmpeg_path:
            raise FileNotFoundError("Could not find ffmpeg executable")
        
        # Construct the ffmpeg command - converting directly to 16kHz, mono
        ffmpeg_cmd = [
            ffmpeg_path,
            '-i', input_file,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', str(target_sr),
            '-y',  # Overwrite output files
            output_file
        ]
        
        # Print the command we're executing
        #print(f"Executing command: {' '.join(ffmpeg_cmd)}")
        
        # Run ffmpeg command and capture output
        process = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the command was successful
        if process.returncode != 0:
            print(f"ffmpeg error: {process.stderr}")
            return False
            
        # Verify output file exists and has content
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            print(f"Output file is empty or doesn't exist: {output_file}")
            return False
            
        #print(f"Successfully converted {input_file} to {output_file}")
        # Delete the input file
        os.remove(input_file)
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        return False
        

if __name__ == "__main__":
    # Base directory containing language folders
    base_dir = "/home/vacl2/multimodal_translation/services/data/languages"
    # List all language directories
    languages = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for lang in tqdm(languages, desc="Languages", unit="lang"):
        lang_dir = os.path.join(base_dir, lang)
        webm_dir = os.path.join(lang_dir, "webm_audio")
        wav_dir = os.path.join(lang_dir, "wav_audio")
        if not os.path.isdir(webm_dir):
            print(f"Warning: {webm_dir} does not exist, skipping.")
            continue
        os.makedirs(wav_dir, exist_ok=True)
        webm_files = [f for f in os.listdir(webm_dir) if f.endswith('.webm')]
        if not webm_files:
            print(f"No .webm files found in {webm_dir}.")
            continue
        for filename in tqdm(webm_files, desc=f"{lang}: Converting files", unit="file"):
            input_file = os.path.join(webm_dir, filename)
            output_file = os.path.join(wav_dir, filename.replace('.webm', '.wav'))
            success = convert_webm_to_wav(input_file, output_file)
            if not success:
                print(f"Failed to convert {input_file} to {output_file}")
