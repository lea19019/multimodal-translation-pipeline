import os
import pandas as pd
import argparse
import random
import subprocess
import shutil
from tqdm import tqdm

# --- New Imports for noisereduce ---
import soundfile as sf
import numpy as np
import noisereduce as nr
import torch
from noisereduce.torchgate import TorchGate as TG

# --- Imports for FFMPEG finder ---
try:
	import imageio_ffmpeg
except ImportError:
	imageio_ffmpeg = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

def find_ffmpeg():
	"""
	Tries to find the ffmpeg binary path in a robust way.
	"""
	# User-provided search paths. You can customize this list for your environment.
	search_paths = [
		"/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/home/myl15/.local/bin/ffmpeg",
		"/home/myl15/.conda/envs/seamless/bin/ffmpeg", "/tmp/ffmpeg-static/ffmpeg"
	]
	for path in search_paths:
		if os.path.exists(path) and os.access(path, os.X_OK): return path
	
	ffmpeg_path = shutil.which("ffmpeg")
	if ffmpeg_path: return ffmpeg_path
	
	if imageio_ffmpeg:
		try: return imageio_ffmpeg.get_ffmpeg_exe()
		except Exception: pass
			
	return None

def process_audio_ffmpeg_only(input_path, output_path, ffmpeg_executable):
	"""
	Processes audio using only FFMPEG for normalization and basic denoising.
	"""
	if not os.path.exists(input_path):
		return False, f"Input file not found: {input_path}"

	command = [
		ffmpeg_executable, '-i', input_path,
		'-af', 'loudnorm=I=-16:TP=-1.5:LRA=11,afftdn',
		'-ar', '16000', '-ac', '1', '-y', output_path
	]
	process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

	if process.returncode != 0:
		return False, f"FFMPEG error on {os.path.basename(input_path)}: {process.stderr.strip()}"
	if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
		return False, f"FFMPEG ran but output file is missing or empty for {os.path.basename(input_path)}"
	
	return True, "Success (ffmpeg only)"

def process_audio_hybrid(input_path, output_path, ffmpeg_executable, temp_dir, tg):
	"""
	Processes audio using a new, improved pipeline:
	1. FFMPEG to standardize format (16kHz, mono).
	2. TorchGate to denoise the standardized audio.
	3. FFMPEG to apply loudness normalization to the clean audio.
	"""
	if not os.path.exists(input_path):
		return False, f"Input file not found: {input_path}"

	# Create paths for a two-step temporary file process
	base_name = os.path.basename(input_path)
	temp_preprocessed_file = os.path.join(temp_dir, f"temp_preprocessed_{base_name}")
	temp_denoised_file = os.path.join(temp_dir, f"temp_denoised_{base_name}")

	try:
		# --- STEP 1: FFMPEG Preprocessing (Resample & Mono ONLY) ---
		# We convert to 16kHz mono first, without touching the volume.
		cmd_preprocess = [
			ffmpeg_executable, '-i', input_path,
			'-ar', '16000', '-ac', '1', '-y', temp_preprocessed_file
		]
		process1 = subprocess.run(cmd_preprocess, capture_output=True, text=True, check=False)
		if process1.returncode != 0:
			return False, f"Preprocessing FFMPEG step failed for {base_name}: {process1.stderr.strip()}"

		# --- STEP 2: Denoising with TorchGate ---
		data, rate = sf.read(temp_preprocessed_file)
		
		# Convert NumPy array to PyTorch Tensor format
		audio_tensor = torch.from_numpy(data.astype(np.float32))
		if audio_tensor.ndim == 1:
			audio_tensor = audio_tensor.unsqueeze(0)
		audio_tensor = audio_tensor.to(device)

		# Apply the model
		enhanced_tensor = tg(audio_tensor)

		# Convert output back to NumPy array
		reduced_noise_data = enhanced_tensor.squeeze(0).cpu().detach().numpy()
		
		# Save the denoised audio to a second temporary file
		sf.write(temp_denoised_file, reduced_noise_data, rate)

		# --- STEP 3: FFMPEG Postprocessing (Loudness Normalization) ---
		# Now, apply loudness normalization to the CLEAN audio.
		cmd_postprocess = [
			ffmpeg_executable, '-i', temp_denoised_file,
			'-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-y', output_path
		]
		process2 = subprocess.run(cmd_postprocess, capture_output=True, text=True, check=False)
		if process2.returncode != 0:
			return False, f"Postprocessing FFMPEG step failed for {base_name}: {process2.stderr.strip()}"
		
		return True, "Success (Denoise -> Normalize)"
		
	except Exception as e:
		import traceback
		tb_str = traceback.format_exc()
		return False, f"torch-gate step failed for {base_name}: {str(e)}\n{tb_str}"
	finally:
		# --- STEP 4: Cleanup ---
		# Clean up both temporary files
		if os.path.exists(temp_preprocessed_file):
			os.remove(temp_preprocessed_file)
		if os.path.exists(temp_denoised_file):
			os.remove(temp_denoised_file)

def main(args):
	"""
	Main function to prepare metadata and process audio.
	"""
	ffmpeg_exe = None
	if not args.no_process:
		print("Locating FFMPEG executable...")
		ffmpeg_exe = find_ffmpeg()
		if not ffmpeg_exe:
			print("\nFATAL ERROR: FFMPEG not found. Aborting.")
			return
		print(f"Found FFMPEG at: {ffmpeg_exe}")

		if args.denoiser == 'noisereduce' and nr is None:
			print("\nFATAL ERROR: --denoiser is set to 'noisereduce', but the library is not installed.")
			print("Please run: pip install noisereduce")
			return

	# Create output directories
	processed_audio_dir = os.path.join(args.output_dir, 'processed_audio_normalized')
	temp_dir = os.path.join(args.output_dir, 'temp_audio')
	if not args.no_process:
		os.makedirs(processed_audio_dir, exist_ok=True)
		if args.denoiser == 'noisereduce':
			os.makedirs(temp_dir, exist_ok=True)
		print(f"Processed audio will be saved in: {processed_audio_dir}")

	# Read the input data file
	# (Existing data reading logic is unchanged)
	if args.input_csv.endswith('.tsv'):
		df = pd.read_csv(args.input_csv, sep='\t', header=None, on_bad_lines='skip')
		df.columns = ['iso_code', 'user_ID', 'source_text', 'target_text', 'filename', 'recording_date', 'recording_duration', 'comment', 'comment_date']
		initial_data = []
		for index, row in df.iterrows():
			lang = str(row.get('iso_code', '')).strip()
			if lang == 'efi': lang = 'efik'
			elif lang == 'ibo': lang = 'igbo'
			elif lang == 'xho': lang = 'xhosa'
			elif lang == 'swa': lang = 'swahili'
			audio_file = f"/home/vacl2/multimodal_translation/services/data/languages/{lang}/wav_audio/{str(row['filename']).strip()}"
			audio_file = audio_file.replace('.webm', '.wav')
			text = str(row.get('target_text', '')).strip().replace('"', '').replace("'", "")
			speaker_name = str(row.get('user_ID', 'unknown')).strip()
			initial_data.append([audio_file, text, speaker_name])
	else:
		df = pd.read_csv(args.input_csv, sep='|', header=None, on_bad_lines='skip')
		df.columns = ['audio_file', 'text', 'normalized_text', 'speaker_name', 'lang']
		initial_data = []
		for index, row in df.iterrows():
			audio_file = str(row.get('audio_file', '')).strip().replace('.webm', '.wav')
			text = str(row.get('text', '')).strip()
			speaker_name = str(row.get('speaker_name', 'unknown')).strip()
			initial_data.append([audio_file, text, speaker_name])


	# --- Processing Step ---
	processed_data = []
	if not args.no_process:
		print(f"\nStarting audio processing using '{args.denoiser}' method...")
		for audio_file, text, speaker_name in tqdm(initial_data, desc="Processing Audio"):
			base_name = os.path.basename(audio_file)
			output_audio_path = os.path.join(processed_audio_dir, base_name)
			
			if args.denoiser == 'noisereduce':
				tg = TG(sr=16000, nonstationary=True, n_std_thresh_stationary=1.2, prop_decrease=0.9, time_mask_smooth_ms=100, freq_mask_smooth_hz=150).to(device)
				success, message = process_audio_hybrid(audio_file, output_audio_path, ffmpeg_exe, temp_dir, tg)
			else: # Default to 'ffmpeg'
				success, message = process_audio_ffmpeg_only(audio_file, output_audio_path, ffmpeg_exe)

			if success:
				processed_data.append([output_audio_path, text, speaker_name])
			else:
				tqdm.write(f"WARNING: Skipping file. {message}")
	else:
		print("Skipping audio processing as per user request.")
		processed_data = initial_data

	# Cleanup temp directory
	if os.path.exists(temp_dir):
		shutil.rmtree(temp_dir)
	
	if not processed_data:
		print("\nError: No data was available to create metadata files. Exiting.")
		return

	# (Existing shuffle, split, and save logic is unchanged)
	random.shuffle(processed_data)
	train_data, eval_data, test_data = [], [], []
	for i, row in enumerate(processed_data):
		if i % 10 == 8: eval_data.append(row)
		elif i % 10 == 9: test_data.append(row)
		else: train_data.append(row)
	columns = ['audio_file', 'text', 'speaker_name']
	train_df = pd.DataFrame(train_data, columns=columns)
	eval_df = pd.DataFrame(eval_data, columns=columns)
	test_df = pd.DataFrame(test_data, columns=columns)
	train_path = os.path.join(args.output_dir, "metadata_train.csv")
	eval_path = os.path.join(args.output_dir, "metadata_eval.csv")
	test_path = os.path.join(args.output_dir, "metadata_test.csv")
	train_df.to_csv(train_path, sep='|', index=False, encoding='utf-8')
	eval_df.to_csv(eval_path, sep='|', index=False, encoding='utf-8')
	test_df.to_csv(test_path, sep='|', index=False, encoding='utf-8')
	
	print("\nScript finished successfully!")
	print(f"Training metadata saved to: {train_path} ({len(train_df)} entries)")
	print(f"Evaluation metadata saved to: {eval_path} ({len(eval_df)} entries)")
	print(f"Test metadata saved to: {test_path} ({len(test_df)} entries)")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prepare metadata and process audio for ASR fine-tuning.")
	parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV/TSV file.")
	parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
	parser.add_argument("--no_process", action="store_true", help="If specified, skips all audio processing.")
	# --- New Argument to choose the denoiser ---
	parser.add_argument(
		"--denoiser",
		type=str,
		choices=['ffmpeg', 'noisereduce', 'none'],
		default='noisereduce',
		help="The denoiser to use'noisereduce' is higher quality. (default: noisereduce)"
	)
	
	args = parser.parse_args()
	
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	main(args)