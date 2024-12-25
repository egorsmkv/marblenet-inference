import os
import logging
import json
from os.path import basename
from typing import Dict, List
from pathlib import Path
from glob import glob
from shutil import rmtree

import sphn
import torch

from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_frame_pred,
    generate_vad_segment_table,
    init_frame_vad_model,
)
from pyannote.core import Annotation
from pyannote.database.util import load_rttm


def as_dict_list(annotation: Annotation) -> Dict[str, List[Dict]]:
    result = {label: [] for label in annotation.labels()}
    for segment, _, label in annotation.itertracks(yield_label=True):
        result[label].append(
            {
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
            }
        )
    return result


logging.getLogger("nemo_logger").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

num_workers = 1
frame_length_in_sec = 0.02

postprocessing = {
    "onset": 0.3,  # onset threshold for detecting the beginning and end of a speech
    "offset": 0.3,  # offset threshold for detecting the end of a speech.
    "pad_onset": 0.2,  # adding durations before each speech segment
    "pad_offset": 0.2,  # adding durations after each speech segment
    "min_duration_on": 0.2,  # threshold for short speech deletion
    "min_duration_off": 0.2,  # threshold for short non-speech segment deletion
    "filter_speech_first": True,
}

vad_model = init_frame_vad_model("vad_multilingual_frame_marblenet")
vad_model = vad_model.to(device)
vad_model.eval()


wavs = glob("wavs/*.wav")

for wav in wavs:
    file_name = basename(wav).replace(".wav", "")
    rttm_filename = f"rttm_outputs/{basename(wav).replace('.wav', '.rttm')}"

    rttm_outputs = Path("rttm_outputs")
    rttm_outputs.mkdir(parents=True, exist_ok=True)

    vad_frame_outputs = Path("vad_frame_outputs")
    vad_frame_outputs.mkdir(parents=True, exist_ok=True)

    tmp_manifest_filename = "tmp.json"

    with open(tmp_manifest_filename, "w") as f:
        durations = sphn.durations([wav])
        row = {
            "audio_filepath": wav,
            "offset": 0,
            "duration": durations[0],
            "label": "infer",
            "text": "-",
        }
        f.write(json.dumps(row) + "\n")

    vad_model.setup_test_data(
        test_data_config={
            "batch_size": 1,
            "sample_rate": 16_000,
            "manifest_filepath": tmp_manifest_filename,
            "labels": ["infer"],
            "num_workers": num_workers,
            "shuffle": False,
            "normalize_audio_db": None,
        }
    )

    pred_dir = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=0.0,
        shift_length_in_sec=0.02,
        manifest_vad_input=tmp_manifest_filename,
        out_dir=str(vad_frame_outputs),
    )

    rttm_out_dir = generate_vad_segment_table(
        vad_pred_dir=pred_dir,
        postprocessing_params=postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=num_workers,
        use_rttm=True,
        out_dir=str(rttm_outputs),
    )

    rttm = load_rttm(rttm_filename)
    speeches = as_dict_list(rttm[file_name])["speech"]
    
    file_chunks_dir = Path(f"chunks/{file_name}")
    if file_chunks_dir.exists():
        rmtree(file_chunks_dir)
    file_chunks_dir.mkdir(parents=True, exist_ok=True)

    reader = sphn.FileReader(wav)

    for idx, speech in enumerate(speeches):
        audio = reader.decode(speech["start"], speech["end"] - speech["start"])

        sphn.write_wav(f"{file_chunks_dir}/{idx}.wav", audio, reader.sample_rate)

        print(speech)

    # Clean up
    os.remove(tmp_manifest_filename)
    rmtree(vad_frame_outputs)
    rmtree(rttm_outputs)
